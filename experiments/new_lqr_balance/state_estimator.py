"""
State Estimator for Reaction Wheel Balance Controller.

Implements sensor fusion and drift compensation:
- IMU (gyro + accelerometer) for body orientation
- Hip encoders for relative leg angles
- Complementary filter with gyro drift compensation

State vector: X = [θ_rel, θ_body, ω_rel, ω_body]^T
"""

import numpy as np
from experiments.new_lqr_balance.config import (
    SensorParams, RobotParams, CONTROL_DT
)


class ComplementaryFilter:
    """
    Complementary filter for angle estimation.
    
    Combines high-frequency gyro data with low-frequency accelerometer data
    to get drift-free angle estimates.
    """
    
    def __init__(self, alpha: float = None, dt: float = CONTROL_DT):
        """
        Initialize complementary filter.
        
        Args:
            alpha: Filter coefficient (0-1). Higher = more gyro trust.
            dt: Sample time
        """
        self.alpha = alpha if alpha is not None else SensorParams.GYRO_WEIGHT
        self.dt = dt
        self.angle = 0.0
    
    def update(self, gyro_rate: float, accel_angle: float) -> float:
        """
        Update angle estimate.
        
        Args:
            gyro_rate: Angular velocity from gyroscope (rad/s)
            accel_angle: Angle computed from accelerometer (rad)
            
        Returns:
            Filtered angle estimate (rad)
        """
        # Gyro integration (high-pass characteristic)
        gyro_angle = self.angle + gyro_rate * self.dt
        
        # Complementary fusion
        self.angle = self.alpha * gyro_angle + (1 - self.alpha) * accel_angle
        
        return self.angle
    
    def reset(self, angle: float = 0.0):
        """Reset filter state."""
        self.angle = angle


class LowPassFilter:
    """Simple first-order low-pass filter."""
    
    def __init__(self, cutoff_hz: float, dt: float = CONTROL_DT):
        """
        Initialize low-pass filter.
        
        Args:
            cutoff_hz: Cutoff frequency in Hz
            dt: Sample time
        """
        self.dt = dt
        self.alpha = self._compute_alpha(cutoff_hz, dt)
        self.value = 0.0
    
    @staticmethod
    def _compute_alpha(cutoff_hz: float, dt: float) -> float:
        """Compute filter coefficient from cutoff frequency."""
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        return dt / (rc + dt)
    
    def update(self, new_value: float) -> float:
        """Update filter with new value."""
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self, value: float = 0.0):
        """Reset filter state."""
        self.value = value


class StateEstimator:
    """
    State estimator for the reaction wheel balance model.
    
    Estimates: X = [θ_rel, θ_body, ω_rel, ω_body]^T
    
    Uses:
    - IMU gyro + accelerometer with drift compensation
    - Hip encoder feedback for θ_rel
    - Differentiation with filtering for velocities
    """
    
    def __init__(self, dt: float = CONTROL_DT):
        """
        Initialize state estimator.
        
        Args:
            dt: Sample time
        """
        self.dt = dt
        
        # State vector
        self.state = np.zeros(4)  # [θ_rel, θ_body, ω_rel, ω_body]
        
        # Complementary filter for body angle
        self.body_filter = ComplementaryFilter(
            alpha=SensorParams.GYRO_WEIGHT, dt=dt
        )
        
        # Low-pass filters for angular velocities
        self.omega_rel_filter = LowPassFilter(
            cutoff_hz=SensorParams.RATE_LPF_CUTOFF, dt=dt
        )
        self.omega_body_filter = LowPassFilter(
            cutoff_hz=SensorParams.RATE_LPF_CUTOFF, dt=dt
        )
        
        # Previous values for velocity estimation
        self.prev_theta_rel = 0.0
        self.prev_theta_body = 0.0
        
        # Gyro drift compensation state
        self.gyro_bias = 0.0
        self.drift_compensation_gain = SensorParams.GYRO_DRIFT_GAIN
        
        # Calibration offsets
        self.baseline_roll = 0.0
        self.baseline_pitch = 0.0
        self.calibrated = False
        
        # Diagonal axis angle for coordinate transformation
        self.psi = np.arctan2(2 * RobotParams.LEG_LR, 2 * RobotParams.LEG_FB)
    
    def calibrate(self, imu_data: dict, num_samples: int = 30):
        """
        Calibrate baseline IMU offsets.
        
        Should be called when robot is standing still in nominal position.
        
        Args:
            imu_data: Current IMU reading (first sample)
            num_samples: Number of samples to average (handled externally)
        """
        self.baseline_roll = imu_data.get('roll', 0.0)
        self.baseline_pitch = imu_data.get('pitch', 0.0)
        self.calibrated = True
    
    def update(self, imu_data: dict, encoder_data: dict) -> np.ndarray:
        """
        Update state estimate from sensor data.
        
        Args:
            imu_data: Dict with 'roll', 'pitch', 'gyro_x', 'gyro_y', 'gyro_z'
            encoder_data: Dict with 'theta_front', 'theta_rear' hip angles
            
        Returns:
            State vector [θ_rel, θ_body, ω_rel, ω_body]
        """
        # === Extract sensor data ===
        roll = imu_data.get('roll', 0.0) - self.baseline_roll
        pitch = imu_data.get('pitch', 0.0) - self.baseline_pitch
        gyro_x = imu_data.get('gyro_x', 0.0)
        gyro_y = imu_data.get('gyro_y', 0.0)
        
        theta_front = encoder_data.get('theta_front', 0.0)
        theta_rear = encoder_data.get('theta_rear', 0.0)
        
        # === Compute relative angle θ_rel ===
        # Average of front and rear hip angles (relative to body)
        theta_rel = (theta_front + theta_rear) / 2.0
        
        # === Compute body inclination θ_body ===
        # Transform roll/pitch to diagonal axis coordinate
        # θ_body is tilt in the plane containing the diagonal support axis
        accel_theta_body = roll * np.cos(self.psi) + pitch * np.sin(self.psi)
        
        # Gyro rate about the diagonal axis
        gyro_body = gyro_x * np.cos(self.psi) + gyro_y * np.sin(self.psi)
        
        # Apply drift compensation: ω_comp = ω_gyro + Kc * θ_rel
        omega_comp = gyro_body + self.drift_compensation_gain * theta_rel
        
        # Complementary filter fusion
        theta_body = self.body_filter.update(omega_comp, accel_theta_body)
        
        # === Compute angular velocities ===
        # ω_rel = d(θ_rel)/dt
        omega_rel_raw = (theta_rel - self.prev_theta_rel) / self.dt
        omega_rel = self.omega_rel_filter.update(omega_rel_raw)
        
        # ω_body from gyro (already have it compensated)
        omega_body = self.omega_body_filter.update(omega_comp)
        
        # Store for next iteration
        self.prev_theta_rel = theta_rel
        self.prev_theta_body = theta_body
        
        # === Update state vector ===
        self.state[0] = theta_rel
        self.state[1] = theta_body
        self.state[2] = omega_rel
        self.state[3] = omega_body
        
        return self.state.copy()
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state.copy()
    
    def get_detailed_state(self) -> dict:
        """
        Get detailed state information for debugging/logging.
        
        Returns:
            Dict with named state components
        """
        return {
            'theta_rel': self.state[0],
            'theta_body': self.state[1],
            'omega_rel': self.state[2],
            'omega_body': self.state[3],
            'theta_body_deg': np.degrees(self.state[1]),
            'theta_rel_deg': np.degrees(self.state[0]),
        }
    
    def reset(self):
        """Reset estimator state."""
        self.state = np.zeros(4)
        self.body_filter.reset()
        self.omega_rel_filter.reset()
        self.omega_body_filter.reset()
        self.prev_theta_rel = 0.0
        self.prev_theta_body = 0.0
        self.gyro_bias = 0.0


class EncoderSimulator:
    """
    Simulates encoder feedback for Mini Pupper.
    
    Since Mini Pupper uses position-controlled servos without direct
    encoder feedback, we estimate joint positions from commanded positions.
    """
    
    def __init__(self, noise_std: float = None):
        """
        Initialize encoder simulator.
        
        Args:
            noise_std: Standard deviation of noise to add (simulates real encoders)
        """
        self.noise_std = noise_std if noise_std is not None else SensorParams.ENCODER_NOISE_STD
        self.theta_front = 0.0
        self.theta_rear = 0.0
    
    def update_from_commanded(self, joint_angles: np.ndarray, 
                               support_legs: list) -> dict:
        """
        Update encoder values from commanded joint angles.
        
        Args:
            joint_angles: 3x4 array of joint angles [abd, hip, knee] x [FR, FL, BR, BL]
            support_legs: List of support leg indices
            
        Returns:
            Dict with 'theta_front', 'theta_rear'
        """
        # Get hip angles of support legs
        front_leg_idx = support_legs[0]
        rear_leg_idx = support_legs[1]
        
        # Hip angle is index 1 in joint_angles
        self.theta_front = joint_angles[1, front_leg_idx]
        self.theta_rear = joint_angles[1, rear_leg_idx]
        
        # Add simulated noise
        if self.noise_std > 0:
            self.theta_front += np.random.normal(0, self.noise_std)
            self.theta_rear += np.random.normal(0, self.noise_std)
        
        return {
            'theta_front': self.theta_front,
            'theta_rear': self.theta_rear,
        }
    
    def get_encoder_data(self) -> dict:
        """Get current encoder values."""
        return {
            'theta_front': self.theta_front,
            'theta_rear': self.theta_rear,
        }
