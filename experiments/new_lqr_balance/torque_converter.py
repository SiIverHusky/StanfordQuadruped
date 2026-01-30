"""
Torque to Position Converter for Mini Pupper Servos.

The Mini Pupper uses position-controlled servos, not torque-controlled actuators.
This module converts desired torques to position commands.

Two approaches:
1. Impedance-based: Δθ = τ / k (position shift proportional to torque)
2. Dynamics-based: Estimate required position to achieve desired torque through 
   servo stiffness

For the reaction wheel model, we need to create "virtual" torque by
commanding position changes that cause the servo's internal PD controller
to produce the desired torque.
"""

import numpy as np
from experiments.new_lqr_balance.config import (
    RobotParams, ControlParams, SafetyParams, CONTROL_DT
)


class ServoModel:
    """
    Model of Mini Pupper servo dynamics.
    
    The servos have internal PD position control:
        τ = Kp * (θ_cmd - θ_actual) + Kd * (0 - θ_dot_actual)
    
    We can command position to achieve desired torque.
    """
    
    # Approximate servo parameters (need calibration for accuracy)
    # These are estimates based on typical hobby servos
    KP_DEFAULT = 3.0    # Position gain (Nm/rad)
    KD_DEFAULT = 0.1    # Velocity gain (Nm·s/rad)
    
    # Servo limits
    MAX_POSITION_RATE = 6.0  # rad/s maximum servo speed
    MAX_TORQUE = 0.5         # Nm maximum torque
    
    def __init__(self, kp: float = None, kd: float = None):
        """
        Initialize servo model.
        
        Args:
            kp: Position gain (Nm/rad)
            kd: Velocity damping (Nm·s/rad)
        """
        self.kp = kp if kp is not None else self.KP_DEFAULT
        self.kd = kd if kd is not None else self.KD_DEFAULT
    
    def torque_to_position_delta(self, desired_torque: float,
                                  current_velocity: float = 0.0) -> float:
        """
        Compute position delta to achieve desired torque.
        
        From τ = Kp * Δθ + Kd * (-θ_dot):
            Δθ = (τ + Kd * θ_dot) / Kp
        
        Args:
            desired_torque: Target torque (Nm)
            current_velocity: Current joint velocity (rad/s)
            
        Returns:
            Position delta to command (rad)
        """
        # Account for velocity damping
        velocity_contribution = self.kd * current_velocity
        
        # Position delta needed
        delta_theta = (desired_torque + velocity_contribution) / self.kp
        
        # Rate limit
        max_delta = self.MAX_POSITION_RATE * CONTROL_DT
        delta_theta = np.clip(delta_theta, -max_delta, max_delta)
        
        return delta_theta
    
    def estimate_torque(self, position_error: float, 
                        velocity: float = 0.0) -> float:
        """
        Estimate torque produced by servo.
        
        Args:
            position_error: θ_cmd - θ_actual (rad)
            velocity: Joint velocity (rad/s)
            
        Returns:
            Estimated torque (Nm)
        """
        torque = self.kp * position_error - self.kd * velocity
        return np.clip(torque, -self.MAX_TORQUE, self.MAX_TORQUE)


class TorqueToPositionConverter:
    """
    Converts desired hip torques to position commands for Mini Pupper.
    
    Strategy:
    1. Maintain a "virtual position" that represents accumulated torque commands
    2. Use servo stiffness model to convert torque to position offset
    3. Apply to the base stance position
    
    This creates an impedance-like behavior where the leg "pushes" with
    the desired torque against the ground.
    """
    
    def __init__(self):
        """Initialize converter."""
        self.servo_model = ServoModel()
        
        # Virtual position offsets (accumulated from torque commands)
        self.front_offset = 0.0
        self.rear_offset = 0.0
        
        # Position rate limits
        self.max_rate = ServoModel.MAX_POSITION_RATE
        
        # Integration decay (prevents windup)
        self.decay_rate = 0.98
        
        # Previous torques for derivative estimation
        self.prev_front_torque = 0.0
        self.prev_rear_torque = 0.0
    
    def convert(self, u_front: float, u_rear: float,
                current_joint_velocities: dict = None) -> dict:
        """
        Convert desired torques to position deltas.
        
        Args:
            u_front: Desired front hip torque (Nm)
            u_rear: Desired rear hip torque (Nm)
            current_joint_velocities: Dict with 'omega_front', 'omega_rear'
            
        Returns:
            Dict with 'delta_front', 'delta_rear' position changes (rad)
        """
        if current_joint_velocities is None:
            current_joint_velocities = {'omega_front': 0.0, 'omega_rear': 0.0}
        
        omega_front = current_joint_velocities.get('omega_front', 0.0)
        omega_rear = current_joint_velocities.get('omega_rear', 0.0)
        
        # Convert torques to position deltas using servo model
        delta_front = self.servo_model.torque_to_position_delta(u_front, omega_front)
        delta_rear = self.servo_model.torque_to_position_delta(u_rear, omega_rear)
        
        # Update accumulated offsets with decay
        self.front_offset = self.decay_rate * self.front_offset + delta_front
        self.rear_offset = self.decay_rate * self.rear_offset + delta_rear
        
        # Store for next iteration
        self.prev_front_torque = u_front
        self.prev_rear_torque = u_rear
        
        return {
            'delta_front': delta_front,
            'delta_rear': delta_rear,
            'offset_front': self.front_offset,
            'offset_rear': self.rear_offset,
        }
    
    def apply_to_stance(self, base_stance: np.ndarray,
                        position_offsets: dict,
                        support_legs: list) -> np.ndarray:
        """
        Apply position offsets to stance matrix.
        
        The hip joint controls the leg pitch, which affects foot position
        in the X-Z plane (forward/back and up/down).
        
        For balance control, we want to shift the effective ground reaction
        force, which is achieved by moving the foot position.
        
        Args:
            base_stance: 3x4 stance matrix [x, y, z] x [FR, FL, BR, BL]
            position_offsets: Output from convert()
            support_legs: List of support leg indices
            
        Returns:
            Modified stance matrix
        """
        stance = base_stance.copy()
        
        front_leg_idx = support_legs[0]
        rear_leg_idx = support_legs[1]
        
        # Get current leg height
        height = abs(base_stance[2, front_leg_idx])
        
        # Hip pitch affects foot X position
        # Δx ≈ height * sin(Δθ_hip) ≈ height * Δθ_hip for small angles
        
        delta_front = position_offsets.get('offset_front', 0.0)
        delta_rear = position_offsets.get('offset_rear', 0.0)
        
        # Apply to X position (forward/backward)
        stance[0, front_leg_idx] += height * delta_front
        stance[0, rear_leg_idx] += height * delta_rear
        
        # Small Z adjustment to maintain ground contact
        # Δz ≈ -height * (1 - cos(Δθ)) ≈ -height * Δθ²/2
        # (usually negligible for small angles)
        
        return stance
    
    def apply_to_joint_angles(self, base_angles: np.ndarray,
                               position_offsets: dict,
                               support_legs: list) -> np.ndarray:
        """
        Apply position offsets directly to joint angles.
        
        This is the more direct approach - modify the hip angles directly.
        
        Args:
            base_angles: 3x4 joint angles [abd, hip, knee] x [FR, FL, BR, BL]
            position_offsets: Output from convert()
            support_legs: List of support leg indices
            
        Returns:
            Modified joint angles
        """
        angles = base_angles.copy()
        
        front_leg_idx = support_legs[0]
        rear_leg_idx = support_legs[1]
        
        # Hip joint is index 1
        angles[1, front_leg_idx] += position_offsets.get('offset_front', 0.0)
        angles[1, rear_leg_idx] += position_offsets.get('offset_rear', 0.0)
        
        return angles
    
    def get_estimated_torques(self, position_errors: dict,
                              velocities: dict) -> dict:
        """
        Estimate actual torques being produced by servos.
        
        Useful for debugging and logging.
        
        Args:
            position_errors: Dict with 'error_front', 'error_rear'
            velocities: Dict with 'omega_front', 'omega_rear'
            
        Returns:
            Dict with estimated torques
        """
        tau_front = self.servo_model.estimate_torque(
            position_errors.get('error_front', 0.0),
            velocities.get('omega_front', 0.0)
        )
        tau_rear = self.servo_model.estimate_torque(
            position_errors.get('error_rear', 0.0),
            velocities.get('omega_rear', 0.0)
        )
        
        return {
            'tau_front': tau_front,
            'tau_rear': tau_rear,
        }
    
    def reset(self):
        """Reset converter state."""
        self.front_offset = 0.0
        self.rear_offset = 0.0
        self.prev_front_torque = 0.0
        self.prev_rear_torque = 0.0
