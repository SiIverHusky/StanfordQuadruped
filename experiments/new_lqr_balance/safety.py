"""
Safety Monitor for Reaction Wheel Balance Controller.

Implements:
- Torque saturation with anti-windup
- Fall detection
- Safe-stand recovery
- Sensor/actuator data logging
"""

import numpy as np
import time
from experiments.new_lqr_balance.config import SafetyParams, CONTROL_DT


class SafetyMonitor:
    """
    Safety monitoring and fall detection.
    
    Monitors:
    - Body tilt angles
    - Angular velocities
    - Control torque magnitudes
    
    Triggers safe-stand if thresholds are exceeded.
    """
    
    def __init__(self):
        """Initialize safety monitor."""
        self.fall_detected = False
        self.safety_triggered = False
        self.last_trigger_time = 0.0
        self.start_time = time.time()
        
        # Running statistics for anomaly detection
        self.tilt_history = []
        self.max_history_length = 100
        
        # Limits from config
        self.max_tilt = SafetyParams.MAX_BODY_TILT
        self.max_relative = SafetyParams.MAX_RELATIVE_ANGLE
        self.max_rate = SafetyParams.MAX_ANGULAR_RATE
        self.fall_threshold = SafetyParams.FALL_TILT_THRESHOLD
        self.cooldown = SafetyParams.SAFETY_RESET_COOLDOWN
    
    def check_state(self, state: np.ndarray) -> dict:
        """
        Check state against safety limits.
        
        Args:
            state: State vector [θ_rel, θ_body, ω_rel, ω_body]
            
        Returns:
            Dict with safety status and any warnings
        """
        theta_rel, theta_body, omega_rel, omega_body = state
        
        current_time = time.time()
        warnings = []
        
        # Track tilt history
        self.tilt_history.append(abs(theta_body))
        if len(self.tilt_history) > self.max_history_length:
            self.tilt_history.pop(0)
        
        # Check body tilt
        if abs(theta_body) > self.fall_threshold:
            self.fall_detected = True
            return {
                'safe': False,
                'fall_detected': True,
                'action': 'emergency_stop',
                'message': f'Fall detected: tilt = {np.degrees(theta_body):.1f}°'
            }
        
        # Check for excessive tilt
        if abs(theta_body) > self.max_tilt:
            warnings.append(f'High tilt: {np.degrees(theta_body):.1f}°')
            
            # Trigger safety if sustained
            if len(self.tilt_history) >= 10:
                recent_avg = np.mean(self.tilt_history[-10:])
                if recent_avg > self.max_tilt:
                    if current_time - self.last_trigger_time > self.cooldown:
                        self.safety_triggered = True
                        self.last_trigger_time = current_time
                        return {
                            'safe': False,
                            'fall_detected': False,
                            'action': 'reduce_gain',
                            'message': 'Sustained high tilt - reducing gains'
                        }
        
        # Check relative angle
        if abs(theta_rel) > self.max_relative:
            warnings.append(f'High relative angle: {np.degrees(theta_rel):.1f}°')
        
        # Check angular velocity
        if abs(omega_body) > self.max_rate:
            warnings.append(f'High angular rate: {omega_body:.2f} rad/s')
        
        return {
            'safe': True,
            'fall_detected': False,
            'action': None,
            'warnings': warnings,
            'message': 'OK' if not warnings else '; '.join(warnings)
        }
    
    def check_control(self, torque: float) -> dict:
        """
        Check control torque.
        
        Args:
            torque: Commanded torque (Nm)
            
        Returns:
            Dict with control status
        """
        if abs(torque) >= SafetyParams.TORQUE_LIMIT:
            return {
                'saturated': True,
                'torque': torque,
                'limit': SafetyParams.TORQUE_LIMIT,
                'message': 'Torque saturated'
            }
        
        return {
            'saturated': False,
            'torque': torque,
            'message': 'OK'
        }
    
    def reset(self):
        """Reset safety monitor."""
        self.fall_detected = False
        self.safety_triggered = False
        self.tilt_history = []


class TorqueLimiter:
    """
    Torque limiter with anti-windup.
    
    Implements smooth saturation to prevent actuator damage
    and integrator windup in the controller.
    """
    
    def __init__(self, max_torque: float = None):
        """
        Initialize torque limiter.
        
        Args:
            max_torque: Maximum allowed torque (Nm)
        """
        self.max_torque = max_torque if max_torque is not None else SafetyParams.TORQUE_LIMIT
        
        # Smooth saturation parameters
        self.soft_limit = self.max_torque * 0.9  # Start softening at 90%
        
        # Statistics
        self.saturation_count = 0
        self.total_calls = 0
    
    def limit(self, torque: float) -> tuple:
        """
        Apply torque limit.
        
        Args:
            torque: Desired torque (Nm)
            
        Returns:
            Tuple of (limited_torque, is_saturated)
        """
        self.total_calls += 1
        
        abs_torque = abs(torque)
        
        # Hard limit
        if abs_torque > self.max_torque:
            self.saturation_count += 1
            return np.sign(torque) * self.max_torque, True
        
        # Soft saturation (smooth transition)
        if abs_torque > self.soft_limit:
            # Quadratic soft saturation
            excess = abs_torque - self.soft_limit
            range_size = self.max_torque - self.soft_limit
            softened = self.soft_limit + range_size * (1 - (1 - excess/range_size)**2)
            return np.sign(torque) * softened, False
        
        return torque, False
    
    def get_statistics(self) -> dict:
        """Get saturation statistics."""
        if self.total_calls == 0:
            return {'saturation_rate': 0.0}
        
        return {
            'saturation_rate': self.saturation_count / self.total_calls,
            'saturation_count': self.saturation_count,
            'total_calls': self.total_calls
        }


class DataLogger:
    """
    Logger for sensor and actuator data.
    
    Logs:
    - State estimates
    - Control outputs
    - Safety events
    - Timing information
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize data logger.
        
        Args:
            max_entries: Maximum log entries before rollover
        """
        self.max_entries = max_entries
        self.data = []
        self.start_time = time.time()
    
    def log(self, entry: dict):
        """
        Log an entry.
        
        Args:
            entry: Dict with data to log
        """
        entry['timestamp'] = time.time() - self.start_time
        self.data.append(entry)
        
        if len(self.data) > self.max_entries:
            self.data.pop(0)
    
    def log_state(self, state: np.ndarray, imu_data: dict, encoder_data: dict):
        """Log state estimation data."""
        self.log({
            'type': 'state',
            'theta_rel': state[0],
            'theta_body': state[1],
            'omega_rel': state[2],
            'omega_body': state[3],
            'imu_roll': imu_data.get('roll', 0),
            'imu_pitch': imu_data.get('pitch', 0),
            'theta_front': encoder_data.get('theta_front', 0),
            'theta_rear': encoder_data.get('theta_rear', 0),
        })
    
    def log_control(self, base_torque: float, u_front: float, u_rear: float,
                    position_offsets: dict):
        """Log control output data."""
        self.log({
            'type': 'control',
            'base_torque': base_torque,
            'u_front': u_front,
            'u_rear': u_rear,
            'offset_front': position_offsets.get('offset_front', 0),
            'offset_rear': position_offsets.get('offset_rear', 0),
        })
    
    def log_safety(self, status: dict):
        """Log safety event."""
        self.log({
            'type': 'safety',
            **status
        })
    
    def save_to_file(self, filename: str):
        """Save log data to CSV file."""
        import csv
        
        if not self.data:
            return
        
        # Get all unique keys
        all_keys = set()
        for entry in self.data:
            all_keys.update(entry.keys())
        
        all_keys = sorted(all_keys)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self.data)
        
        print(f"[Logger] Saved {len(self.data)} entries to {filename}")
    
    def get_summary(self) -> dict:
        """Get summary statistics from logged data."""
        if not self.data:
            return {}
        
        state_entries = [e for e in self.data if e.get('type') == 'state']
        control_entries = [e for e in self.data if e.get('type') == 'control']
        
        summary = {
            'total_entries': len(self.data),
            'duration': self.data[-1].get('timestamp', 0) if self.data else 0,
        }
        
        if state_entries:
            theta_body_vals = [e['theta_body'] for e in state_entries if 'theta_body' in e]
            if theta_body_vals:
                summary['max_tilt_deg'] = np.degrees(max(abs(v) for v in theta_body_vals))
                summary['avg_tilt_deg'] = np.degrees(np.mean([abs(v) for v in theta_body_vals]))
        
        if control_entries:
            torque_vals = [e['base_torque'] for e in control_entries if 'base_torque' in e]
            if torque_vals:
                summary['max_torque'] = max(abs(v) for v in torque_vals)
                summary['avg_torque'] = np.mean([abs(v) for v in torque_vals])
        
        return summary
    
    def clear(self):
        """Clear all logged data."""
        self.data = []
        self.start_time = time.time()
