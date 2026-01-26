"""
6-DOF Model Decomposition for Diagonal Balance.

Implements the decoupling from Section 3 of the paper (Figure 7).

The 6-DOF diagonal leg model is divided into three independent parts:
- Part A: Posture control around diagonal axis (LQR - the main balance problem)
- Part B: Control along diagonal axis (PID)
- Part C: Height control (PID)

This decomposition simplifies the control design by treating each part separately.
"""

import numpy as np
from experiments.lqr_diagonal_balance.config import (
    RobotParams, LQRParams, PartBParams, PartCParams,
    CONTROL_DT, DIAGONAL_AXIS_ANGLE
)
from experiments.lqr_diagonal_balance.dynamics import DiagonalDynamics, ZYKAngles
from experiments.lqr_diagonal_balance.lqr_controller import (
    LQRController, KalmanFilter, PIDController
)
from experiments.lqr_diagonal_balance.vmc import VirtualModelControl


class PartAController:
    """
    Part A: Posture control around diagonal axis.
    
    This is the main balance controller using LQR.
    
    From Section 3 of the paper:
    - Controls torques of joints rotating around the diagonal line
    - Resists postural disturbance (tipping over the diagonal axis)
    - Uses the biped supporting model dynamics
    """
    
    def __init__(self):
        """Initialize Part A controller."""
        # Dynamics model
        self.dynamics = DiagonalDynamics()
        
        # LQR controller
        self.lqr = LQRController(self.dynamics)
        
        # Kalman filter for state estimation
        self.kalman = KalmanFilter(self.dynamics)
        
        # Z-Y-K angle converter
        self.zyk = ZYKAngles(DIAGONAL_AXIS_ANGLE)
        
        # Reference state (upright)
        self.reference = np.zeros(6)
        
        # Last control output
        self.last_torque = np.zeros(2)
    
    def reset(self):
        """Reset controller state."""
        self.kalman.reset()
        self.last_torque = np.zeros(2)
    
    def update(self, imu_data: dict, joint_angles: dict, dt: float) -> np.ndarray:
        """
        Compute Part A control output.
        
        Args:
            imu_data: Dict with 'roll', 'pitch', 'yaw', 'gyro_*'
            joint_angles: Dict with joint angles for support legs
            dt: Time step
            
        Returns:
            Virtual torques [τ_AF, τ_AH] for diagonal-axis joints
        """
        # Convert IMU data to Z-Y-K representation
        theta_z, theta_y, theta_k = self.zyk.euler_to_zyk(
            imu_data.get('roll', 0),
            imu_data.get('pitch', 0),
            imu_data.get('yaw', 0)
        )
        
        # Construct measurement vector
        # For the biped model: θ_F = front leg angle, θ_H = hind leg angle
        # We approximate these from body orientation and joint angles
        theta_body = theta_k  # Body rotation about diagonal
        
        # Leg angles relative to vertical (approximate from joint angles)
        theta_F = theta_body + joint_angles.get('front_pitch', 0)
        theta_H = theta_body + joint_angles.get('hind_pitch', 0)
        
        z = np.array([theta_F, theta_H, theta_body])
        
        # Kalman filter update
        self.kalman.predict(self.last_torque, dt)
        self.kalman.update(z)
        
        # Get estimated state
        state = self.kalman.get_state_estimate()
        
        # Compute LQR control
        u = self.lqr.compute_control(state, self.reference)
        
        self.last_torque = u
        
        return u
    
    def get_stability_info(self) -> dict:
        """Get LQR stability information."""
        return self.lqr.get_stability_info()


class PartBController:
    """
    Part B: Control along diagonal axis.
    
    From Section 3 of the paper:
    - Controls torques of joints perpendicular to diagonal
    - Resists disturbance along the diagonal line
    - Uses simple PID control (no dynamic balance involved)
    """
    
    def __init__(self):
        """Initialize Part B controller."""
        self.pid = PIDController(
            kp=PartBParams.KP,
            ki=PartBParams.KI,
            kd=PartBParams.KD,
            output_min=-PartBParams.MAX_TORQUE,
            output_max=PartBParams.MAX_TORQUE,
            integral_max=PartBParams.INTEGRAL_MAX
        )
        
        # Target: zero rotation perpendicular to diagonal
        self.setpoint = 0.0
    
    def reset(self):
        """Reset controller state."""
        self.pid.reset()
    
    def update(self, imu_data: dict, dt: float) -> float:
        """
        Compute Part B control output.
        
        Args:
            imu_data: Dict with 'roll', 'pitch', 'yaw'
            dt: Time step
            
        Returns:
            Torque for perpendicular control
        """
        # Compute rotation perpendicular to diagonal
        roll = imu_data.get('roll', 0)
        pitch = imu_data.get('pitch', 0)
        
        # Project onto perpendicular axis
        cos_psi = np.cos(DIAGONAL_AXIS_ANGLE)
        sin_psi = np.sin(DIAGONAL_AXIS_ANGLE)
        theta_perp = -roll * sin_psi + pitch * cos_psi
        
        # Compute error
        error = self.setpoint - theta_perp
        
        # PID control
        torque = self.pid.update(error, dt)
        
        return torque


class PartCController:
    """
    Part C: Height control.
    
    From Section 3 of the paper:
    - Controls forces of prismatic joints (leg extension)
    - Maintains desired robot height
    - Uses simple PID control
    
    For Mini Pupper, this translates to controlling the knee joints.
    """
    
    def __init__(self):
        """Initialize Part C controller."""
        self.pid = PIDController(
            kp=PartCParams.KP,
            ki=PartCParams.KI,
            kd=PartCParams.KD,
            output_min=-PartCParams.MAX_FORCE,
            output_max=PartCParams.MAX_FORCE,
            integral_max=PartCParams.INTEGRAL_MAX
        )
        
        # Target height
        self.target_height = PartCParams.TARGET_HEIGHT
    
    def reset(self):
        """Reset controller state."""
        self.pid.reset()
    
    def update(self, current_height: float, dt: float) -> float:
        """
        Compute Part C control output.
        
        Args:
            current_height: Estimated current body height (m)
            dt: Time step
            
        Returns:
            Force command for height adjustment
        """
        # Height error
        error = self.target_height - current_height
        
        # PID control
        force = self.pid.update(error, dt)
        
        return force
    
    def set_target_height(self, height: float):
        """Set target height."""
        self.target_height = height


class SixDOFDecomposition:
    """
    Complete 6-DOF decomposition controller.
    
    Combines Part A, B, and C into a unified control interface.
    
    From Section 3 (Figure 7):
    - Decouples the 6-DOF model into three independent control problems
    - Each part can be tuned and tested independently
    - VMC transforms outputs to actual joint commands
    """
    
    def __init__(self):
        """Initialize the decomposed controller."""
        # Individual controllers
        self.part_a = PartAController()  # LQR for diagonal balance
        self.part_b = PartBController()  # PID for along-diagonal
        self.part_c = PartCController()  # PID for height
        
        # Virtual model control for mapping to actual joints
        self.vmc = VirtualModelControl()
        
        # Control timing
        self.dt = CONTROL_DT
        self.last_time = None
        
        # Output storage
        self.last_output = {
            'part_a_torques': np.zeros(2),
            'part_b_torque': 0.0,
            'part_c_force': 0.0,
            'combined_torques': np.zeros(4),
        }
    
    def reset(self):
        """Reset all controllers."""
        self.part_a.reset()
        self.part_b.reset()
        self.part_c.reset()
        self.last_time = None
    
    def update(self, imu_data: dict, joint_angles: dict, 
               body_height: float, dt: float = None) -> dict:
        """
        Compute complete control output.
        
        Args:
            imu_data: IMU readings dict with roll, pitch, yaw, gyro data
            joint_angles: Current joint angles for support legs
            body_height: Estimated body height (m)
            dt: Time step (uses config default if None)
            
        Returns:
            Dict with control outputs for each part and combined torques
        """
        if dt is None:
            dt = self.dt
        
        # Part A: LQR balance control
        tau_a = self.part_a.update(imu_data, joint_angles, dt)
        
        # Part B: Along-diagonal control
        tau_b = self.part_b.update(imu_data, dt)
        
        # Part C: Height control
        f_c = self.part_c.update(body_height, dt)
        
        # Combine outputs
        # Part A gives virtual torques, transform to normal joints
        state = {
            'roll_F': joint_angles.get('front_roll', 0),
            'pitch_F': joint_angles.get('front_pitch', 0),
            'roll_H': joint_angles.get('hind_roll', 0),
            'pitch_H': joint_angles.get('hind_pitch', 0),
        }
        
        # Transform Part A virtual torques to joint torques
        tau_joints = self.vmc.virtual_to_normal_torques(tau_a, state)
        
        # Add Part B contribution (perpendicular to diagonal)
        # This affects both roll and pitch in a coupled way
        cos_psi = np.cos(DIAGONAL_AXIS_ANGLE)
        sin_psi = np.sin(DIAGONAL_AXIS_ANGLE)
        
        tau_joints[0] -= tau_b * sin_psi  # Front roll
        tau_joints[1] += tau_b * cos_psi  # Front pitch
        tau_joints[2] -= tau_b * sin_psi  # Hind roll
        tau_joints[3] += tau_b * cos_psi  # Hind pitch
        
        # Part C affects knee joints (converted from force to torque)
        # For position control, this will affect the z-height target
        height_adjustment = f_c / 100.0  # Approximate conversion
        
        # Store outputs
        self.last_output = {
            'part_a_torques': tau_a.copy(),
            'part_b_torque': tau_b,
            'part_c_force': f_c,
            'combined_torques': tau_joints.copy(),
            'height_adjustment': height_adjustment,
        }
        
        return self.last_output
    
    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about all controllers.
        
        Returns:
            Dict with stability info and current outputs
        """
        return {
            'part_a_stability': self.part_a.get_stability_info(),
            'last_output': self.last_output.copy(),
        }


class BalanceStateEstimator:
    """
    Estimates the robot's balance state from sensor data.
    
    Combines IMU data and joint angles to estimate:
    - Body orientation relative to diagonal axis
    - Body height
    - Stability margin
    """
    
    def __init__(self):
        """Initialize state estimator."""
        self.params = RobotParams()
        self.vmc = VirtualModelControl()
        
        # Filtered values
        self.filtered_height = self.params.STANDING_HEIGHT
        self.height_filter_alpha = 0.1
    
    def estimate_state(self, imu_data: dict, joint_angles: dict,
                       support_legs: list) -> dict:
        """
        Estimate complete balance state.
        
        Args:
            imu_data: IMU readings
            joint_angles: All joint angles (12 values)
            support_legs: Indices of legs currently supporting
            
        Returns:
            Dict with estimated state values
        """
        # Transform IMU to virtual coordinates
        virtual_imu = self.vmc.transform_imu_to_virtual(
            imu_data.get('roll', 0),
            imu_data.get('pitch', 0),
            imu_data.get('yaw', 0)
        )
        
        # Estimate body height from support leg configuration
        height = self._estimate_height(joint_angles, support_legs)
        
        # Filter height
        self.filtered_height = (
            self.height_filter_alpha * height +
            (1 - self.height_filter_alpha) * self.filtered_height
        )
        
        # Compute stability margin (simplified)
        stability = self._compute_stability_margin(virtual_imu)
        
        return {
            'theta_k': virtual_imu['theta_k'],      # Diagonal axis rotation
            'theta_perp': virtual_imu['theta_perp'],# Perpendicular rotation
            'yaw': virtual_imu['theta_yaw'],
            'body_height': self.filtered_height,
            'stability_margin': stability,
            'is_stable': stability > 0.1,
        }
    
    def _estimate_height(self, joint_angles: dict, 
                         support_legs: list) -> float:
        """
        Estimate body height from joint angles.
        
        Uses forward kinematics on support legs.
        
        Args:
            joint_angles: Dict mapping joint names to angles
            support_legs: List of support leg indices
            
        Returns:
            Estimated height in meters
        """
        # Simplified: use nominal height with adjustment from hip pitch
        # Average the support leg configurations
        height_sum = 0.0
        
        for leg_idx in support_legs:
            # Get hip pitch for this leg
            pitch_key = f'leg{leg_idx}_hip'
            hip_pitch = joint_angles.get(pitch_key, 0)
            
            # Get knee angle
            knee_key = f'leg{leg_idx}_knee'
            knee_angle = joint_angles.get(knee_key, 0)
            
            # Forward kinematics (simplified)
            L1 = self.params.LEG_L1
            L2 = self.params.LEG_L2
            
            # Height contribution from this leg
            h = L1 * np.cos(hip_pitch) + L2 * np.cos(hip_pitch + knee_angle)
            height_sum += abs(h)
        
        if len(support_legs) > 0:
            return height_sum / len(support_legs)
        else:
            return self.params.STANDING_HEIGHT
    
    def _compute_stability_margin(self, virtual_imu: dict) -> float:
        """
        Compute simplified stability margin.
        
        Based on how close we are to tipping over.
        
        Args:
            virtual_imu: Virtual coordinate IMU data
            
        Returns:
            Stability margin (0 to 1, higher is more stable)
        """
        # Maximum safe angle - be generous for diagonal balance
        # During diagonal stance, 20-30° tilt is expected and acceptable
        max_angle = np.deg2rad(45)
        
        # Current tilt magnitude
        theta_k = abs(virtual_imu['theta_k'])
        theta_perp = abs(virtual_imu['theta_perp'])
        
        # Combined tilt
        total_tilt = np.sqrt(theta_k**2 + theta_perp**2)
        
        # Stability as fraction of max angle remaining
        stability = max(0, 1 - total_tilt / max_angle)
        
        return stability
