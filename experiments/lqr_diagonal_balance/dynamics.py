"""
Dynamics Model for Diagonal Leg Balance.

Implements the Lagrangian dynamics from the paper (Section 2):
- Biped supporting model with two legs connected to a trunk
- State-space formulation for LQR control

The dynamics are derived using the Lagrange method with:
- Kinetic energy (Ek) from body and leg motion
- Potential energy (Ep) from gravity
"""

import numpy as np
from experiments.lqr_diagonal_balance.config import RobotParams


class DiagonalDynamics:
    """
    Lagrangian dynamics for the diagonal leg supporting model.
    
    Based on equations (3)-(7) from the paper.
    
    State vector: X = [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
    where:
        θ_F = front leg angle from vertical
        θ_H = hind leg angle from vertical
        θ_body = body pitch angle
        
    Control input: U = [τ_F, τ_H]
    where:
        τ_F = torque at front hip
        τ_H = torque at hind hip
    """
    
    def __init__(self, params: RobotParams = None):
        """
        Initialize dynamics model.
        
        Args:
            params: Robot physical parameters (uses defaults if None)
        """
        if params is None:
            params = RobotParams()
        
        # Store parameters
        self.m_body = params.BODY_MASS
        self.m_leg = params.LEG_MASS + params.MODULE_MASS / 2  # Include hip module
        self.W = params.HALF_WIDTH
        self.D = params.DIAGONAL_DISTANCE
        self.L_leg = params.LEG_LENGTH * 0.7  # Effective leg length
        self.g = params.GRAVITY
        
        # Friction for torque limiting
        self.mu_static = params.STATIC_FRICTION
        
        # State dimension
        self.n_states = 6
        self.n_inputs = 2
        
        # Precompute mass matrix components
        self._precompute_constants()
    
    def _precompute_constants(self):
        """Precompute constant terms for efficiency."""
        # Total mass
        self.m_total = self.m_body + 2 * self.m_leg
        
        # Moment of inertia approximations
        # Body as rectangular plate rotating about edge
        self.I_body = self.m_body * self.W ** 2 / 3
        
        # Leg as rod rotating about end
        self.I_leg = self.m_leg * self.L_leg ** 2 / 3
    
    def compute_leg_angles(self, theta_FS: float, theta_HS: float, 
                           theta_body: float) -> tuple:
        """
        Compute absolute leg angles from servo angles and body angle.
        
        From equations (1) and (2) in the paper:
            θ_F = θ_body + θ_FS
            θ_H = θ_body + θ_HS
        
        Args:
            theta_FS: Front servo angle (relative to body)
            theta_HS: Hind servo angle (relative to body)
            theta_body: Body pitch angle
            
        Returns:
            Tuple of (theta_F, theta_H) - absolute leg angles from vertical
        """
        theta_F = theta_body + theta_FS
        theta_H = theta_body + theta_HS
        return theta_F, theta_H
    
    def compute_kinetic_energy(self, state: np.ndarray) -> float:
        """
        Compute total kinetic energy (equation 5 in paper).
        
        Ek = (1/2) * sum of all mass elements * velocity^2
        
        Args:
            state: State vector [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
            
        Returns:
            Total kinetic energy (J)
        """
        theta_F, theta_F_dot = state[0], state[1]
        theta_H, theta_H_dot = state[2], state[3]
        theta_body, theta_body_dot = state[4], state[5]
        
        # Body kinetic energy (translation + rotation)
        # Body CoM velocity depends on leg angles
        # Simplified: body rotates about the line between feet
        v_body_sq = (self.L_leg ** 2) * (theta_body_dot ** 2)
        Ek_body = 0.5 * self.m_body * v_body_sq + 0.5 * self.I_body * theta_body_dot ** 2
        
        # Leg kinetic energy
        Ek_front_leg = 0.5 * self.I_leg * theta_F_dot ** 2
        Ek_hind_leg = 0.5 * self.I_leg * theta_H_dot ** 2
        
        return Ek_body + Ek_front_leg + Ek_hind_leg
    
    def compute_potential_energy(self, state: np.ndarray) -> float:
        """
        Compute total potential energy (equation 6 in paper).
        
        Ep = sum of (mass * g * height)
        
        Args:
            state: State vector [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
            
        Returns:
            Total potential energy (J)
        """
        theta_F = state[0]
        theta_H = state[2]
        theta_body = state[4]
        
        # Heights of mass centers
        # Front leg CoM at half leg length
        h_front_leg = 0.5 * self.L_leg * np.cos(theta_F)
        
        # Hind leg CoM at half leg length
        h_hind_leg = 0.5 * self.L_leg * np.cos(theta_H)
        
        # Body CoM (simplified as midpoint between hips at leg length)
        h_body = self.L_leg * np.cos(theta_body) - self.W * np.sin(theta_body)
        
        Ep = self.g * (
            self.m_body * h_body +
            self.m_leg * h_front_leg +
            self.m_leg * h_hind_leg
        )
        
        return Ep
    
    def compute_mass_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the mass/inertia matrix M(q).
        
        For the Lagrangian dynamics: M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = τ
        
        Args:
            state: State vector
            
        Returns:
            3x3 mass matrix for [θ_F, θ_H, θ_body]
        """
        theta_F = state[0]
        theta_H = state[2]
        theta_body = state[4]
        
        # Simplified mass matrix (diagonal dominant for well-separated legs)
        M = np.zeros((3, 3))
        
        # Diagonal terms
        M[0, 0] = self.I_leg + self.m_leg * self.L_leg ** 2 / 4
        M[1, 1] = self.I_leg + self.m_leg * self.L_leg ** 2 / 4
        M[2, 2] = self.I_body + self.m_body * self.L_leg ** 2
        
        # Coupling terms (body couples to both legs)
        coupling = 0.5 * self.m_leg * self.L_leg ** 2 * np.cos(theta_body)
        M[0, 2] = M[2, 0] = coupling * np.cos(theta_F - theta_body)
        M[1, 2] = M[2, 1] = coupling * np.cos(theta_H - theta_body)
        
        return M
    
    def compute_gravity_vector(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the gravity vector G(q).
        
        Args:
            state: State vector
            
        Returns:
            3x1 gravity vector for [θ_F, θ_H, θ_body]
        """
        theta_F = state[0]
        theta_H = state[2]
        theta_body = state[4]
        
        G = np.zeros(3)
        
        # Gravity terms (derivative of potential energy)
        G[0] = -0.5 * self.m_leg * self.g * self.L_leg * np.sin(theta_F)
        G[1] = -0.5 * self.m_leg * self.g * self.L_leg * np.sin(theta_H)
        G[2] = -self.m_body * self.g * (
            self.L_leg * np.sin(theta_body) + 
            self.W * np.cos(theta_body)
        )
        
        return G
    
    def linearize(self, equilibrium_state: np.ndarray = None) -> tuple:
        """
        Linearize the dynamics around an equilibrium point.
        
        Produces state-space form: x_dot = A*x + B*u
        
        Based on equation (8) in the paper.
        
        Args:
            equilibrium_state: State to linearize around (default: upright)
            
        Returns:
            Tuple of (A, B) state-space matrices
        """
        if equilibrium_state is None:
            # Upright equilibrium: all angles zero
            equilibrium_state = np.zeros(6)
        
        # Extract positions
        theta_F_eq = equilibrium_state[0]
        theta_H_eq = equilibrium_state[2]
        theta_body_eq = equilibrium_state[4]
        
        # Compute linearized matrices
        # State: [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
        
        # A matrix (6x6)
        A = np.zeros((6, 6))
        
        # Velocity terms (derivative of position = velocity)
        A[0, 1] = 1.0  # θ_F_dot
        A[2, 3] = 1.0  # θ_H_dot
        A[4, 5] = 1.0  # θ_body_dot
        
        # Acceleration terms (from linearized dynamics)
        # Simplified: dominant terms are gravity restoring forces
        
        # Front leg acceleration depends on its angle (pendulum effect)
        A[1, 0] = self.m_leg * self.g * self.L_leg / (2 * self.I_leg)
        
        # Hind leg acceleration
        A[3, 2] = self.m_leg * self.g * self.L_leg / (2 * self.I_leg)
        
        # Body acceleration (inverted pendulum)
        A[5, 4] = self.m_body * self.g * self.L_leg / self.I_body
        
        # Coupling: body angle affects leg accelerations and vice versa
        coupling_strength = 0.1  # Small coupling for linearized model
        A[1, 4] = coupling_strength * A[1, 0]
        A[3, 4] = coupling_strength * A[3, 2]
        A[5, 0] = coupling_strength * A[5, 4]
        A[5, 2] = coupling_strength * A[5, 4]
        
        # B matrix (6x2) - torque inputs affect angular accelerations
        B = np.zeros((6, 2))
        
        # Torque on front hip accelerates front leg and body
        B[1, 0] = 1.0 / self.I_leg  # Front leg
        B[5, 0] = -1.0 / self.I_body  # Reaction on body
        
        # Torque on hind hip accelerates hind leg and body
        B[3, 1] = 1.0 / self.I_leg  # Hind leg
        B[5, 1] = -1.0 / self.I_body  # Reaction on body
        
        return A, B
    
    def compute_max_torque(self, state: np.ndarray) -> float:
        """
        Compute maximum allowable hip torque before slippage (equation 10).
        
        τ_max = μ_static * F * L_leg
        
        where F is the normal force on the leg.
        
        Args:
            state: Current state vector
            
        Returns:
            Maximum torque (Nm)
        """
        theta_body = state[4]
        
        # Normal force on each leg (simplified: half the total weight)
        # Adjusted for body tilt
        F_normal = 0.5 * self.m_total * self.g * np.cos(theta_body)
        
        # Maximum friction force
        F_friction_max = self.mu_static * F_normal
        
        # Maximum torque
        tau_max = F_friction_max * self.L_leg
        
        return tau_max
    
    def state_derivative(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute state derivative for simulation.
        
        Args:
            state: Current state [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
            u: Control input [τ_F, τ_H]
            
        Returns:
            State derivative
        """
        # Get linearized dynamics
        A, B = self.linearize(state)
        
        # Compute derivative
        x_dot = A @ state + B @ u
        
        return x_dot
    
    def simulate_step(self, state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate one time step using RK4 integration.
        
        Args:
            state: Current state
            u: Control input
            dt: Time step
            
        Returns:
            New state
        """
        # RK4 integration
        k1 = self.state_derivative(state, u)
        k2 = self.state_derivative(state + 0.5 * dt * k1, u)
        k3 = self.state_derivative(state + 0.5 * dt * k2, u)
        k4 = self.state_derivative(state + dt * k3, u)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return new_state


class ZYKAngles:
    """
    Z-Y-K angle representation for the 6-DOF model.
    
    From Section 3 of the paper (equations 11-14):
    - Standard Z-Y-X Euler angles but with the third rotation around the K-axis
    - K-axis is along the diagonal line connecting supporting legs
    
    This allows body rotation to be expressed relative to the support diagonal.
    """
    
    def __init__(self, diagonal_angle: float):
        """
        Initialize Z-Y-K angle converter.
        
        Args:
            diagonal_angle: Angle of diagonal axis relative to X-axis (radians)
        """
        self.psi = diagonal_angle  # K-axis angle
    
    def rotation_matrix_zyk(self, theta_z: float, theta_y: float, 
                            theta_k: float) -> np.ndarray:
        """
        Compute rotation matrix for Z-Y-K angles (equation 11).
        
        R = R_z(θ_z) * R_y(θ_y) * R_k(θ_k)
        
        where R_k is rotation about the diagonal axis.
        
        Args:
            theta_z: Yaw angle
            theta_y: Pitch angle
            theta_k: Rotation about diagonal axis
            
        Returns:
            3x3 rotation matrix
        """
        # Z rotation
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])
        
        # Y rotation
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        
        # K-axis rotation (combined X-Y based on diagonal angle)
        c_psi = np.cos(self.psi)
        s_psi = np.sin(self.psi)
        c_k = np.cos(theta_k)
        s_k = np.sin(theta_k)
        
        # Rodrigues rotation formula for arbitrary axis
        # K-axis direction: [cos(psi), sin(psi), 0]
        K = np.array([c_psi, s_psi, 0])
        K_skew = np.array([
            [0, -K[2], K[1]],
            [K[2], 0, -K[0]],
            [-K[1], K[0], 0]
        ])
        
        Rk = np.eye(3) + s_k * K_skew + (1 - c_k) * (K_skew @ K_skew)
        
        # Combined rotation
        R = Rz @ Ry @ Rk
        
        return R
    
    def euler_to_zyk(self, roll: float, pitch: float, yaw: float) -> tuple:
        """
        Convert standard Euler angles (roll, pitch, yaw) to Z-Y-K angles.
        
        This is needed because IMU typically gives Euler angles but we need
        the rotation about the diagonal axis for Part A control.
        
        Args:
            roll: Roll angle (X-axis rotation)
            pitch: Pitch angle (Y-axis rotation)
            yaw: Yaw angle (Z-axis rotation)
            
        Returns:
            Tuple of (theta_z, theta_y, theta_k)
        """
        # First compute the rotation matrix from Euler angles
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        # Extract Z-Y-K angles from rotation matrix
        # theta_y from R[2,0]
        theta_y = np.arcsin(-R[2, 0])
        
        # theta_z from R[1,0]/cos(theta_y) and R[0,0]/cos(theta_y)
        c_y = np.cos(theta_y)
        if abs(c_y) > 1e-6:
            theta_z = np.arctan2(R[1, 0] / c_y, R[0, 0] / c_y)
        else:
            theta_z = 0
        
        # theta_k from the remaining rotation about K-axis
        # Project the roll component onto the diagonal axis
        theta_k = roll * np.cos(self.psi) + pitch * np.sin(self.psi)
        
        return theta_z, theta_y, theta_k
    
    def zyk_to_euler(self, theta_z: float, theta_y: float, 
                     theta_k: float) -> tuple:
        """
        Convert Z-Y-K angles back to standard Euler angles.
        
        Args:
            theta_z: Yaw (Z rotation)
            theta_y: Pitch-like (Y rotation)
            theta_k: Diagonal axis rotation
            
        Returns:
            Tuple of (roll, pitch, yaw)
        """
        # Decompose theta_k into roll and pitch components
        roll = theta_k * np.cos(self.psi)
        pitch = theta_y + theta_k * np.sin(self.psi)
        yaw = theta_z
        
        return roll, pitch, yaw
