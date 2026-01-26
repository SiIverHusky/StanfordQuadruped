"""
LQR Controller for Diagonal Balance.

Implements the LQR control law from Section 2 of the paper (equations 8-9).

The controller computes optimal state feedback gains K such that:
    u = -K * x

where x is the state vector and u is the control input (hip torques).
"""

import numpy as np
from scipy import linalg
from experiments.lqr_diagonal_balance.config import LQRParams, RobotParams
from experiments.lqr_diagonal_balance.dynamics import DiagonalDynamics


class LQRController:
    """
    Linear Quadratic Regulator for diagonal leg balance.
    
    Based on equation (9) in the paper:
        u = -K * x
    
    The gain matrix K is computed by solving the continuous-time
    algebraic Riccati equation (CARE).
    """
    
    def __init__(self, dynamics: DiagonalDynamics = None,
                 Q: np.ndarray = None, R: np.ndarray = None):
        """
        Initialize LQR controller.
        
        Args:
            dynamics: Dynamics model (creates default if None)
            Q: State cost matrix (uses LQRParams.Q if None)
            R: Control cost matrix (uses LQRParams.R if None)
        """
        self.dynamics = dynamics if dynamics is not None else DiagonalDynamics()
        self.Q = Q if Q is not None else LQRParams.Q
        self.R = R if R is not None else LQRParams.R
        
        # Torque limits
        self.max_torque = LQRParams.MAX_HIP_TORQUE
        
        # State-space matrices (will be computed)
        self.A = None
        self.B = None
        self.K = None
        self.P = None  # Solution to Riccati equation
        
        # Compute gains for upright equilibrium
        self.compute_gains()
    
    def compute_gains(self, equilibrium_state: np.ndarray = None):
        """
        Compute LQR gains for a given equilibrium point.
        
        Solves the continuous-time algebraic Riccati equation:
            A'P + PA - PBR^{-1}B'P + Q = 0
        
        Then computes the feedback gain:
            K = R^{-1} B' P
        
        Args:
            equilibrium_state: State to linearize around (default: upright)
        """
        # Get linearized dynamics
        self.A, self.B = self.dynamics.linearize(equilibrium_state)
        
        # Solve CARE using scipy
        try:
            self.P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            
            # Compute gain matrix
            self.K = np.linalg.inv(self.R) @ self.B.T @ self.P
            
        except np.linalg.LinAlgError as e:
            print(f"[LQR] Warning: Riccati equation solve failed: {e}")
            print("[LQR] Using fallback PD gains")
            self._use_fallback_gains()
    
    def _use_fallback_gains(self):
        """
        Use simple PD-like gains if CARE solution fails.
        
        This provides a stable baseline even if the optimal solution
        cannot be computed.
        """
        # Hand-tuned gains based on the structure of the problem
        self.K = np.zeros((2, 6))
        
        # Front leg control: respond to front leg angle and body angle
        self.K[0, 0] = 10.0   # Front leg angle
        self.K[0, 1] = 2.0    # Front leg velocity
        self.K[0, 4] = 5.0    # Body angle
        self.K[0, 5] = 1.0    # Body velocity
        
        # Hind leg control: respond to hind leg angle and body angle
        self.K[1, 2] = 10.0   # Hind leg angle
        self.K[1, 3] = 2.0    # Hind leg velocity
        self.K[1, 4] = 5.0    # Body angle
        self.K[1, 5] = 1.0    # Body velocity
    
    def compute_control(self, state: np.ndarray, 
                        reference: np.ndarray = None) -> np.ndarray:
        """
        Compute control input using LQR feedback law.
        
        u = -K * (x - x_ref)
        
        Args:
            state: Current state vector [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
            reference: Reference state (default: zeros = upright)
            
        Returns:
            Control input [τ_F, τ_H] (hip torques in Nm)
        """
        if reference is None:
            reference = np.zeros(6)
        
        # State error
        error = state - reference
        
        # LQR control law
        u = -self.K @ error
        
        # Apply torque limits
        u = self._limit_torque(u, state)
        
        return u
    
    def _limit_torque(self, u: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Apply torque limits to prevent slippage.
        
        From equation (10) in the paper:
            τ_max = μ_static * F * L_leg
        
        Args:
            u: Proposed control input
            state: Current state (for computing max torque)
            
        Returns:
            Limited control input
        """
        # Get maximum torque from dynamics
        tau_max = self.dynamics.compute_max_torque(state)
        
        # Also apply hardware limit
        tau_max = min(tau_max, self.max_torque)
        
        # Clip torques
        u_limited = np.clip(u, -tau_max, tau_max)
        
        return u_limited
    
    def get_stability_info(self) -> dict:
        """
        Get information about closed-loop stability.
        
        Returns:
            Dictionary with stability metrics
        """
        if self.A is None or self.K is None:
            return {"stable": False, "message": "Gains not computed"}
        
        # Closed-loop A matrix
        A_cl = self.A - self.B @ self.K
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvals(A_cl)
        
        # Check stability (all eigenvalues have negative real part)
        stable = np.all(np.real(eigenvalues) < 0)
        
        # Dominant eigenvalue (slowest mode)
        dominant_idx = np.argmax(np.real(eigenvalues))
        dominant_eigenvalue = eigenvalues[dominant_idx]
        
        return {
            "stable": stable,
            "eigenvalues": eigenvalues,
            "dominant_eigenvalue": dominant_eigenvalue,
            "settling_time_approx": -4.0 / np.real(dominant_eigenvalue) if np.real(dominant_eigenvalue) < 0 else np.inf,
            "gain_matrix": self.K,
        }


class KalmanFilter:
    """
    Kalman filter for state estimation with noisy sensors.
    
    From Section 5 of the paper:
    - Joint angle sensors: std = 0.001 rad
    - Posture angle sensors: std = 0.005 rad
    
    The filter estimates the full state from noisy measurements.
    """
    
    def __init__(self, dynamics: DiagonalDynamics = None,
                 Q_process: np.ndarray = None,
                 R_measurement: np.ndarray = None):
        """
        Initialize Kalman filter.
        
        Args:
            dynamics: Dynamics model
            Q_process: Process noise covariance
            R_measurement: Measurement noise covariance
        """
        self.dynamics = dynamics if dynamics is not None else DiagonalDynamics()
        
        # Dimensions
        self.n_states = 6
        self.n_measurements = 3  # θ_F, θ_H, θ_body (positions only)
        
        # Get linearized dynamics
        self.A, self.B = self.dynamics.linearize()
        
        # Measurement matrix (observe positions only)
        self.C = np.array([
            [1, 0, 0, 0, 0, 0],  # θ_F
            [0, 0, 1, 0, 0, 0],  # θ_H
            [0, 0, 0, 0, 1, 0],  # θ_body
        ])
        
        # Noise covariances
        if Q_process is None:
            # Process noise (model uncertainty)
            self.Q = np.diag([0.001, 0.01, 0.001, 0.01, 0.001, 0.01])
        else:
            self.Q = Q_process
        
        if R_measurement is None:
            # Measurement noise (sensor noise)
            self.R = np.diag([0.001**2, 0.001**2, 0.005**2])
        else:
            self.R = R_measurement
        
        # State estimate and covariance
        self.x_hat = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 0.1
    
    def predict(self, u: np.ndarray, dt: float):
        """
        Prediction step of Kalman filter.
        
        Args:
            u: Control input
            dt: Time step
        """
        # Discrete-time state transition
        A_d = np.eye(self.n_states) + self.A * dt
        B_d = self.B * dt
        
        # State prediction
        self.x_hat = A_d @ self.x_hat + B_d @ u
        
        # Covariance prediction
        Q_d = self.Q * dt
        self.P = A_d @ self.P @ A_d.T + Q_d
    
    def update(self, z: np.ndarray):
        """
        Update step of Kalman filter with new measurement.
        
        Args:
            z: Measurement vector [θ_F, θ_H, θ_body]
        """
        # Innovation
        y = z - self.C @ self.x_hat
        
        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R
        
        # Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        # State update
        self.x_hat = self.x_hat + K @ y
        
        # Covariance update
        I = np.eye(self.n_states)
        self.P = (I - K @ self.C) @ self.P
    
    def get_state_estimate(self) -> np.ndarray:
        """
        Get current state estimate.
        
        Returns:
            Estimated state vector
        """
        return self.x_hat.copy()
    
    def reset(self, initial_state: np.ndarray = None):
        """
        Reset filter to initial state.
        
        Args:
            initial_state: Initial state estimate (default: zeros)
        """
        if initial_state is None:
            self.x_hat = np.zeros(self.n_states)
        else:
            self.x_hat = initial_state.copy()
        
        self.P = np.eye(self.n_states) * 0.1


class PIDController:
    """
    Simple PID controller for Part B and Part C control.
    
    These simpler controllers are used for:
    - Part B: Along-diagonal disturbance (doesn't involve dynamic balance)
    - Part C: Height maintenance (straightforward position control)
    """
    
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_max: float = 1.0):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            integral_max: Maximum integral accumulation
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        
        # State
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    
    def update(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error (setpoint - measurement)
            dt: Time step
            
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        self.last_error = error
        
        # Total output
        output = p_term + i_term + d_term
        
        # Clip to limits
        output = np.clip(output, self.output_min, self.output_max)
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
