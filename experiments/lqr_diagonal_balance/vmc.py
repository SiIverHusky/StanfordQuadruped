"""
Virtual Model Control (VMC) for Diagonal Balance.

Implements the VMC approach from Section 4 of the paper (equations 15-23).

VMC provides a bridge between:
- Virtual model: joints rotate around/perpendicular to the diagonal axis
- Normal model: typical quadruped with roll/pitch joints

This allows us to:
1. Design a controller for the simpler virtual model
2. Map the control outputs to the actual robot joints
"""

import numpy as np
from experiments.lqr_diagonal_balance.config import RobotParams, DIAGONAL_AXIS_ANGLE


class VirtualModelControl:
    """
    Virtual Model Control transformation.
    
    From Section 4 of the paper:
    - Virtual model has joints aligned with the diagonal axis
    - Normal model has standard roll/pitch joints
    - VMC transforms torques between the two representations
    
    The procedure:
    1. Map sensor information from normal model to virtual model
    2. Use virtual model controller to compute virtual torques
    3. Map virtual torques back to normal model joints
    """
    
    def __init__(self, diagonal_angle: float = None, leg_height: float = None):
        """
        Initialize VMC transformer.
        
        Args:
            diagonal_angle: Angle of diagonal axis (radians), default from config
            leg_height: Standing leg height (m), default from config
        """
        self.psi = diagonal_angle if diagonal_angle is not None else DIAGONAL_AXIS_ANGLE
        self.H = leg_height if leg_height is not None else RobotParams.STANDING_HEIGHT
        
        # Robot parameters
        self.params = RobotParams()
        
        # Precompute trigonometric values
        self.cos_psi = np.cos(self.psi)
        self.sin_psi = np.sin(self.psi)
    
    def virtual_to_normal_angles(self, theta_AF: float, theta_AH: float) -> tuple:
        """
        Convert virtual model angles to normal model angles.
        
        From equations (15-18) in the paper.
        
        Virtual model (around diagonal):
            - theta_AF: Front leg angle around diagonal axis
            - theta_AH: Hind leg angle around diagonal axis
        
        Normal model (roll/pitch):
            - theta_roll: Hip roll (abduction)
            - theta_pitch: Hip pitch
        
        Args:
            theta_AF: Virtual front leg angle (rad)
            theta_AH: Virtual hind leg angle (rad)
            
        Returns:
            Tuple of ((roll_F, pitch_F), (roll_H, pitch_H))
        """
        # From equation (17):
        # For front leg, the virtual angle projects onto roll and pitch
        roll_F = theta_AF * self.cos_psi
        pitch_F = theta_AF * self.sin_psi
        
        # From equation (18):
        # For hind leg
        roll_H = theta_AH * self.cos_psi
        pitch_H = theta_AH * self.sin_psi
        
        return (roll_F, pitch_F), (roll_H, pitch_H)
    
    def normal_to_virtual_angles(self, roll_F: float, pitch_F: float,
                                  roll_H: float, pitch_H: float) -> tuple:
        """
        Convert normal model angles to virtual model angles.
        
        Inverse of virtual_to_normal_angles.
        
        Args:
            roll_F: Front leg roll angle
            pitch_F: Front leg pitch angle
            roll_H: Hind leg roll angle
            pitch_H: Hind leg pitch angle
            
        Returns:
            Tuple of (theta_AF, theta_AH)
        """
        # Project roll/pitch onto diagonal axis direction
        theta_AF = roll_F * self.cos_psi + pitch_F * self.sin_psi
        theta_AH = roll_H * self.cos_psi + pitch_H * self.sin_psi
        
        return theta_AF, theta_AH
    
    def compute_virtual_jacobian(self, theta_AF: float) -> np.ndarray:
        """
        Compute force Jacobian for virtual model (equation 19).
        
        J_virtual = dX/dq for the virtual model
        
        Args:
            theta_AF: Virtual leg angle
            
        Returns:
            2x1 Jacobian matrix [dX/dθ, dY/dθ]
        """
        # From equation (19):
        # Position along diagonal: r = H * sin(theta_AF)
        # Position perpendicular: (constant for this DOF)
        J = np.array([
            [self.H * np.cos(theta_AF)],  # dr/dθ along diagonal
            [0]  # No perpendicular motion for this DOF
        ])
        
        return J
    
    def compute_normal_jacobian(self, theta_roll: float, 
                                 theta_pitch: float) -> np.ndarray:
        """
        Compute force Jacobian for normal model (equation 20).
        
        J_normal = dX/dq for the normal roll/pitch model
        
        Args:
            theta_roll: Roll joint angle
            theta_pitch: Pitch joint angle
            
        Returns:
            2x2 Jacobian matrix
        """
        # From equation (20):
        J = np.array([
            [self.H * np.cos(theta_roll), 0],
            [0, self.H * np.cos(theta_pitch)]
        ])
        
        return J
    
    def virtual_to_normal_torques(self, tau_virtual: np.ndarray,
                                   state: dict) -> np.ndarray:
        """
        Transform virtual model torques to normal model torques.
        
        From equations (21-23) in the paper:
            τ_normal = M * τ_virtual
        
        where M is the transformation matrix.
        
        Args:
            tau_virtual: Virtual torques [τ_AF, τ_AH]
            state: Current joint state dict with roll/pitch angles
            
        Returns:
            Normal torques [τ_roll_F, τ_pitch_F, τ_roll_H, τ_pitch_H]
        """
        # Get current angles
        roll_F = state.get('roll_F', 0.0)
        pitch_F = state.get('pitch_F', 0.0)
        roll_H = state.get('roll_H', 0.0)
        pitch_H = state.get('pitch_H', 0.0)
        
        # Compute virtual angles for Jacobian computation
        theta_AF, theta_AH = self.normal_to_virtual_angles(
            roll_F, pitch_F, roll_H, pitch_H
        )
        
        # Compute Jacobians
        J_virtual_F = self.compute_virtual_jacobian(theta_AF)
        J_virtual_H = self.compute_virtual_jacobian(theta_AH)
        J_normal_F = self.compute_normal_jacobian(roll_F, pitch_F)
        J_normal_H = self.compute_normal_jacobian(roll_H, pitch_H)
        
        # From equation (22-23):
        # The transformation distributes diagonal torque to roll/pitch
        # based on the diagonal axis orientation
        
        # Front leg transformation
        tau_roll_F = tau_virtual[0] * self.cos_psi
        tau_pitch_F = tau_virtual[0] * self.sin_psi
        
        # Hind leg transformation
        tau_roll_H = tau_virtual[1] * self.cos_psi
        tau_pitch_H = tau_virtual[1] * self.sin_psi
        
        return np.array([tau_roll_F, tau_pitch_F, tau_roll_H, tau_pitch_H])
    
    def normal_to_virtual_torques(self, tau_normal: np.ndarray) -> np.ndarray:
        """
        Transform normal model torques to virtual model torques.
        
        Inverse of virtual_to_normal_torques (for sensing/logging).
        
        Args:
            tau_normal: Normal torques [τ_roll_F, τ_pitch_F, τ_roll_H, τ_pitch_H]
            
        Returns:
            Virtual torques [τ_AF, τ_AH]
        """
        # Project torques onto diagonal axis
        tau_AF = tau_normal[0] * self.cos_psi + tau_normal[1] * self.sin_psi
        tau_AH = tau_normal[2] * self.cos_psi + tau_normal[3] * self.sin_psi
        
        return np.array([tau_AF, tau_AH])
    
    def transform_imu_to_virtual(self, roll: float, pitch: float, 
                                  yaw: float) -> dict:
        """
        Transform IMU readings to virtual model coordinates.
        
        The IMU gives roll/pitch/yaw in the body frame.
        We need to express this relative to the diagonal axis.
        
        Args:
            roll: Body roll angle from IMU
            pitch: Body pitch angle from IMU
            yaw: Body yaw angle from IMU
            
        Returns:
            Dict with virtual model angles
        """
        # Rotation about diagonal axis (theta_k in paper's Z-Y-K angles)
        # This is the key angle for Part A control
        theta_k = roll * self.cos_psi + pitch * self.sin_psi
        
        # Rotation perpendicular to diagonal (for Part B control)
        theta_perp = -roll * self.sin_psi + pitch * self.cos_psi
        
        return {
            'theta_k': theta_k,        # About diagonal (Part A)
            'theta_perp': theta_perp,  # Perpendicular (Part B)
            'theta_yaw': yaw,          # Yaw (affects both)
            'roll': roll,              # Original values for reference
            'pitch': pitch,
        }
    
    def compute_foot_positions_virtual(self, theta_AF: float, theta_AH: float,
                                       height: float = None) -> tuple:
        """
        Compute foot positions in virtual model coordinates.
        
        Args:
            theta_AF: Front leg virtual angle
            theta_AH: Hind leg virtual angle
            height: Leg height (default: standing height)
            
        Returns:
            Tuple of (front_foot_pos, hind_foot_pos) as 3D vectors
        """
        if height is None:
            height = self.H
        
        # Front foot position (in diagonal coordinates)
        front_x = height * np.sin(theta_AF) * self.cos_psi
        front_y = height * np.sin(theta_AF) * self.sin_psi
        front_z = -height * np.cos(theta_AF)
        
        # Hind foot position
        hind_x = height * np.sin(theta_AH) * self.cos_psi
        hind_y = height * np.sin(theta_AH) * self.sin_psi
        hind_z = -height * np.cos(theta_AH)
        
        # Add leg origin offsets (diagonal distance)
        D_half = self.params.DIAGONAL_DISTANCE / 2
        
        front_pos = np.array([
            front_x + D_half * self.cos_psi,
            front_y + D_half * self.sin_psi,
            front_z
        ])
        
        hind_pos = np.array([
            hind_x - D_half * self.cos_psi,
            hind_y - D_half * self.sin_psi,
            hind_z
        ])
        
        return front_pos, hind_pos


class TorqueToJointMapper:
    """
    Maps computed torques to actual Mini Pupper joint commands.
    
    The Mini Pupper uses position control, not direct torque control.
    This class converts desired torques to joint position adjustments.
    """
    
    def __init__(self):
        """Initialize the mapper."""
        self.params = RobotParams()
        
        # Approximate joint stiffness (Nm/rad)
        # This converts torque commands to position adjustments
        self.joint_stiffness = {
            'abduction': 0.5,  # Roll joints
            'hip': 1.0,        # Pitch joints
            'knee': 0.8,       # Knee joints
        }
        
        # Position rate limits (rad/s)
        self.max_rate = {
            'abduction': 2.0,
            'hip': 3.0,
            'knee': 3.0,
        }
    
    def torque_to_position_delta(self, torques: np.ndarray, 
                                  dt: float) -> np.ndarray:
        """
        Convert desired torques to position deltas.
        
        For position-controlled servos, we simulate torque control by:
        Δθ = τ / k
        
        where k is the effective stiffness.
        
        Args:
            torques: Desired torques [τ_roll_F, τ_pitch_F, τ_roll_H, τ_pitch_H]
            dt: Time step
            
        Returns:
            Position deltas [Δroll_F, Δpitch_F, Δroll_H, Δpitch_H]
        """
        # Convert torques to position adjustments
        deltas = np.zeros(4)
        
        # Front roll (abduction)
        deltas[0] = torques[0] / self.joint_stiffness['abduction']
        
        # Front pitch (hip)
        deltas[1] = torques[1] / self.joint_stiffness['hip']
        
        # Hind roll (abduction)
        deltas[2] = torques[2] / self.joint_stiffness['abduction']
        
        # Hind pitch (hip)
        deltas[3] = torques[3] / self.joint_stiffness['hip']
        
        # Apply rate limits
        max_delta_abd = self.max_rate['abduction'] * dt
        max_delta_hip = self.max_rate['hip'] * dt
        
        deltas[0] = np.clip(deltas[0], -max_delta_abd, max_delta_abd)
        deltas[1] = np.clip(deltas[1], -max_delta_hip, max_delta_hip)
        deltas[2] = np.clip(deltas[2], -max_delta_abd, max_delta_abd)
        deltas[3] = np.clip(deltas[3], -max_delta_hip, max_delta_hip)
        
        return deltas
    
    def apply_to_stance(self, stance: np.ndarray, 
                        position_deltas: np.ndarray,
                        support_legs: list) -> np.ndarray:
        """
        Apply position adjustments to stance matrix.
        
        Only modifies the support legs (the ones doing the balancing).
        
        Args:
            stance: 3x4 stance matrix [x, y, z] for each leg
            position_deltas: Joint position deltas from torque_to_position_delta
            support_legs: List of leg indices that are supporting
            
        Returns:
            Modified stance matrix
        """
        new_stance = stance.copy()
        
        # The position deltas affect foot position through kinematics
        # Simplified: roll affects Y position, pitch affects X/Z
        
        for i, leg_idx in enumerate(support_legs):
            if i == 0:  # Front support leg
                roll_delta = position_deltas[0]
                pitch_delta = position_deltas[1]
            else:  # Hind support leg
                roll_delta = position_deltas[2]
                pitch_delta = position_deltas[3]
            
            # Approximate effect on foot position
            # Roll rotates foot in Y-Z plane
            height = abs(stance[2, leg_idx])
            new_stance[1, leg_idx] += height * np.sin(roll_delta)
            
            # Pitch rotates foot in X-Z plane
            new_stance[0, leg_idx] += height * np.sin(pitch_delta)
        
        return new_stance
