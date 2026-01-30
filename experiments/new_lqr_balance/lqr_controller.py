"""
LQR Controller for Reaction Wheel Inverted Pendulum Model.

Implements the state-feedback control law:
    u = -k^T X

where:
    X = [θ_rel, θ_body, ω_rel, ω_body]^T
    u = base torque demand for hip joints

With cooperative torque distribution:
    u_c = Kf*(θ_front − θ_rel) + Kr*(θ_rear − θ_rel)
    u_front = (u / 2) + u_c
    u_rear  = (u / 2) − u_c
"""

import numpy as np
from scipy import linalg
from experiments.new_lqr_balance.config import (
    RobotParams, ControlParams, SafetyParams, CONTROL_DT
)


class ReactionWheelDynamics:
    """
    Dynamics model for reaction wheel inverted pendulum.
    
    Models:
    - Body + lifted legs as reaction wheel
    - Supporting legs as pendulum
    
    Linearized about vertical equilibrium for LQR design.
    """
    
    def __init__(self):
        """Initialize dynamics model."""
        # Physical parameters
        self.m_body = RobotParams.PENDULUM_MASS
        self.m_wheel = RobotParams.WHEEL_MASS
        self.L = RobotParams.PENDULUM_LENGTH
        self.g = RobotParams.GRAVITY
        
        # Moments of inertia
        self.I_body = RobotParams.pendulum_inertia()
        self.I_wheel = RobotParams.wheel_inertia()
        
        # Total inertia about pivot
        self.I_total = self.I_body + self.I_wheel
    
    def linearize(self) -> tuple:
        """
        Linearize dynamics about vertical equilibrium.
        
        State: X = [θ_rel, θ_body, ω_rel, ω_body]
        Input: u = τ (hip torque)
        
        Linearized dynamics:
            X_dot = A @ X + B @ u
        
        Returns:
            Tuple of (A, B) state-space matrices
        """
        # Shorthand
        m = self.m_body
        L = self.L
        g = self.g
        I = self.I_total
        Iw = self.I_wheel
        
        # Linearized A matrix
        # The reaction wheel pendulum has coupled dynamics:
        # - θ_body'' depends on gravity (m*g*L*sin(θ_body) ≈ m*g*L*θ_body)
        # - θ_rel affects the system through reaction torques
        
        # State vector: [θ_rel, θ_body, ω_rel, ω_body]
        A = np.zeros((4, 4))
        
        # Kinematics: d/dt θ = ω
        A[0, 2] = 1.0  # d/dt θ_rel = ω_rel
        A[1, 3] = 1.0  # d/dt θ_body = ω_body
        
        # Dynamics: d/dt ω = f(θ, u)
        # Gravity torque effect on body angle
        gravity_gain = (m * g * L) / I
        A[3, 1] = gravity_gain  # d/dt ω_body depends on θ_body (unstable pole!)
        
        # Coupling: relative angle affects body through reaction
        A[3, 0] = -gravity_gain * 0.1  # Weak coupling
        
        # Linearized B matrix
        # Torque input affects:
        # - ω_rel directly (we're commanding relative motion)
        # - ω_body through reaction (Newton's 3rd law)
        B = np.zeros((4, 1))
        
        # Torque affects relative angle acceleration
        B[2, 0] = 1.0 / Iw  # τ accelerates wheel relative to body
        
        # Reaction torque on body
        B[3, 0] = -1.0 / I  # Equal and opposite on body (reaction)
        
        return A, B
    
    def compute_nonlinear_dynamics(self, state: np.ndarray, 
                                    torque: float) -> np.ndarray:
        """
        Compute nonlinear state derivative.
        
        Args:
            state: [θ_rel, θ_body, ω_rel, ω_body]
            torque: Applied hip torque
            
        Returns:
            State derivative [θ_rel_dot, θ_body_dot, ω_rel_dot, ω_body_dot]
        """
        theta_rel, theta_body, omega_rel, omega_body = state
        
        # Shorthand
        m = self.m_body
        L = self.L
        g = self.g
        I = self.I_total
        Iw = self.I_wheel
        
        # State derivative
        state_dot = np.zeros(4)
        
        # Kinematics
        state_dot[0] = omega_rel
        state_dot[1] = omega_body
        
        # Dynamics (nonlinear)
        gravity_torque = m * g * L * np.sin(theta_body)
        
        # Body angular acceleration
        state_dot[3] = (gravity_torque - torque) / I
        
        # Relative angular acceleration
        state_dot[2] = torque / Iw
        
        return state_dot


class ReactionWheelLQR:
    """
    LQR controller for reaction wheel balance.
    
    Computes optimal state feedback gain K such that:
        u = -K @ x
    
    Includes cooperative torque distribution for front/rear legs.
    """
    
    def __init__(self, Q: np.ndarray = None, R: np.ndarray = None):
        """
        Initialize LQR controller.
        
        Args:
            Q: State cost matrix (4x4)
            R: Control cost matrix (1x1)
        """
        self.Q = Q if Q is not None else ControlParams.Q
        self.R = R if R is not None else ControlParams.R
        
        # Dynamics model
        self.dynamics = ReactionWheelDynamics()
        
        # LQR gain (computed in compute_gains)
        self.K = None
        self.P = None  # Riccati solution
        
        # Cooperative distribution gains
        self.Kf = ControlParams.KF_COOPERATIVE
        self.Kr = ControlParams.KR_COOPERATIVE
        
        # Torque limits
        self.max_torque = SafetyParams.TORQUE_LIMIT
        
        # Anti-windup integrator
        self.integrator = 0.0
        self.integrator_limit = SafetyParams.INTEGRATOR_LIMIT
        
        # Compute gains
        self.compute_gains()
    
    def compute_gains(self):
        """
        Compute LQR gains by solving the algebraic Riccati equation.
        
        A'P + PA - PBR^{-1}B'P + Q = 0
        K = R^{-1} B' P
        """
        A, B = self.dynamics.linearize()
        
        try:
            # Solve continuous-time algebraic Riccati equation
            self.P = linalg.solve_continuous_are(A, B, self.Q, self.R)
            
            # Compute feedback gain
            self.K = np.linalg.inv(self.R) @ B.T @ self.P
            
            print("[LQR] Gain matrix computed successfully:")
            print(f"  K = {self.K.flatten()}")
            
            # Check closed-loop eigenvalues
            A_cl = A - B @ self.K
            eigvals = np.linalg.eigvals(A_cl)
            print(f"  Closed-loop eigenvalues: {eigvals}")
            
            if np.all(np.real(eigvals) < 0):
                print("  ✓ System is stable")
            else:
                print("  ⚠ Warning: System may be unstable!")
                
        except np.linalg.LinAlgError as e:
            print(f"[LQR] Warning: Riccati solve failed: {e}")
            self._use_fallback_gains()
    
    def _use_fallback_gains(self):
        """Use hand-tuned gains if CARE solution fails."""
        print("[LQR] Using fallback PD-like gains")
        
        # Manually tuned gains based on physical intuition
        # u = -K @ [θ_rel, θ_body, ω_rel, ω_body]
        self.K = np.array([[
            5.0,    # θ_rel: moderate response to relative angle
            50.0,   # θ_body: strong response to body tilt (main stabilization)
            1.0,    # ω_rel: damp relative motion
            10.0,   # ω_body: damp body angular velocity
        ]])
    
    def compute_control(self, state: np.ndarray, 
                        reference: np.ndarray = None) -> float:
        """
        Compute base control torque using LQR feedback.
        
        u = -K @ (x - x_ref)
        
        Args:
            state: Current state [θ_rel, θ_body, ω_rel, ω_body]
            reference: Reference state (default: zeros = vertical)
            
        Returns:
            Base torque demand (Nm)
        """
        if reference is None:
            reference = np.zeros(4)
        
        # State error
        error = state - reference
        
        # LQR control
        u = -float(self.K @ error)
        
        # Apply saturation with anti-windup
        u_saturated = np.clip(u, -self.max_torque, self.max_torque)
        
        # Anti-windup: only integrate when not saturated
        if u_saturated == u:
            self.integrator += error[1] * CONTROL_DT  # Integrate body angle error
            self.integrator = np.clip(
                self.integrator, 
                -self.integrator_limit, 
                self.integrator_limit
            )
        
        # Add integral term (helps with steady-state errors)
        u_saturated += 0.5 * self.integrator
        
        # Final saturation
        return np.clip(u_saturated, -self.max_torque, self.max_torque)
    
    def distribute_torque(self, base_torque: float,
                          encoder_data: dict,
                          theta_rel: float) -> tuple:
        """
        Distribute base torque to front and rear legs using cooperative control.
        
        From the spec:
            u_c = Kf*(θ_front − θ_rel) + Kr*(θ_rear − θ_rel)
            u_front = (u / 2) + u_c
            u_rear  = (u / 2) − u_c
        
        Args:
            base_torque: Total torque demand from LQR
            encoder_data: Dict with 'theta_front', 'theta_rear'
            theta_rel: Current relative angle estimate
            
        Returns:
            Tuple of (u_front, u_rear) torques
        """
        theta_front = encoder_data.get('theta_front', 0.0)
        theta_rear = encoder_data.get('theta_rear', 0.0)
        
        # Cooperative correction term
        # This helps suppress yaw twist by coordinating front and rear
        u_c = (self.Kf * (theta_front - theta_rel) + 
               self.Kr * (theta_rear - theta_rel))
        
        # Distribute torque
        u_front = (base_torque / 2.0) + u_c
        u_rear = (base_torque / 2.0) - u_c
        
        # Apply individual limits
        u_front = np.clip(u_front, -self.max_torque, self.max_torque)
        u_rear = np.clip(u_rear, -self.max_torque, self.max_torque)
        
        return u_front, u_rear
    
    def reset(self):
        """Reset controller state (integrators, etc.)."""
        self.integrator = 0.0
