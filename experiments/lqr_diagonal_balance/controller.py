"""
Main LQR Diagonal Balance Controller.

Integrates all components:
- 6-DOF decomposition (Part A, B, C)
- Virtual Model Control
- Hardware interface
- Safety monitoring

This is the main entry point for running the LQR-based diagonal balance experiment.
"""

import numpy as np
import time
import signal
import sys
import os
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.lqr_diagonal_balance.config import (
    RobotParams, LQRParams, SafetyLimits, Mode,
    CONTROL_DT, TIMEOUT_S, LIFT_TIME, HOLD_TIME, RETURN_TIME,
    SUPPORT_PAIR, DIAGONAL_PAIRS, DEFAULT_STANCE_MATRIX,
    DEBUG_PRINT, LOG_TO_FILE, LOG_FILE
)
from experiments.lqr_diagonal_balance.decomposition import (
    SixDOFDecomposition, BalanceStateEstimator
)
from experiments.lqr_diagonal_balance.vmc import TorqueToJointMapper


class LQRDiagonalBalancer:
    """
    Main controller for LQR-based diagonal balance.
    
    Implements the paper's approach:
    1. LQR control for Part A (diagonal-axis balance)
    2. PID control for Parts B and C
    3. VMC to map to actual joints
    """
    
    def __init__(self, mode: str = Mode.SIMULATION):
        """
        Initialize the balancer.
        
        Args:
            mode: Operating mode (simulation, bench, physical)
        """
        self.mode = mode
        self.running = False
        
        # Controllers
        self.controller = SixDOFDecomposition()
        self.state_estimator = BalanceStateEstimator()
        self.torque_mapper = TorqueToJointMapper()
        
        # Configuration
        self.support_pair = SUPPORT_PAIR
        self.diagonal_config = DIAGONAL_PAIRS[self.support_pair]
        self.support_legs = self.diagonal_config["support"]
        self.lift_legs = self.diagonal_config["lift"]
        
        # State
        self.current_stance = DEFAULT_STANCE_MATRIX.copy()
        self.lift_height = 0.0
        self.target_lift_height = 0.02  # Start with 2cm lift (was 3cm)
        
        # Control flags
        self.enable_active_balance = True  # Enable active balance
        
        # Simple balance gains (meters of foot shift per radian of tilt)
        # Positive tilt -> need to shift feet in compensating direction
        self.balance_gain_x = 0.06  # Forward/back correction
        self.balance_gain_y = 0.06  # Left/right correction
        
        # Rate limiting for smooth movements (max meters per second)
        self.max_correction_rate = 0.15  # 15cm per second max (was 1cm - too slow!)
        self.last_x_correction = 0.0
        self.last_y_correction = 0.0
        
        # IMU baseline calibration (compensate for resting tilt)
        self.baseline_roll = 0.0
        self.baseline_pitch = 0.0
        
        # Timing
        self.start_time = None
        self.last_loop_time = None
        
        # Hardware interfaces (initialized if needed)
        self.hardware = None
        self.imu = None
        self.config_obj = None
        
        # Logging
        self.log_data = []
        self.log_file = None
        
        # Safety
        self.emergency_stop = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n[LQR Balancer] Received interrupt, stopping...")
        self.emergency_stop = True
        self.stop()
    
    def initialize_hardware(self) -> bool:
        """
        Initialize hardware interfaces.
        
        Returns:
            True if successful, False otherwise
        """
        if self.mode == Mode.SIMULATION:
            print("[LQR Balancer] Simulation mode - no hardware")
            return True
        
        try:
            print("[LQR Balancer] Initializing hardware...")
            
            # Import hardware modules
            from MangDang.mini_pupper.HardwareInterface import HardwareInterface
            from MangDang.mini_pupper.Config import Configuration
            from MangDang.mini_pupper.ESP32Interface import ESP32Interface
            from pupper.Kinematics import four_legs_inverse_kinematics
            
            # Store for later
            self._four_legs_ik = four_legs_inverse_kinematics
            
            # Create instances
            self.config_obj = Configuration()
            self.hardware = HardwareInterface()
            self.esp32 = ESP32Interface()
            
            # Note: IMU data is read from ESP32Interface, not separate IMU class
            # The ESP32 has an integrated IMU accessed via esp32.imu_get_data()
            self.imu = None  # Will use esp32.imu_get_data() instead
            
            print("[LQR Balancer] Hardware initialized successfully")
            return True
            
        except Exception as e:
            print(f"[LQR Balancer] Hardware initialization failed: {e}")
            return False
    
    def initialize_logging(self):
        """Initialize data logging."""
        if not LOG_TO_FILE:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lqr_balance_{timestamp}.csv"
        
        self.log_file = open(filename, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        
        # Header
        self.log_writer.writerow([
            'time', 'roll', 'pitch', 'yaw',
            'theta_k', 'theta_perp', 'height', 'stability',
            'tau_a_f', 'tau_a_h', 'tau_b', 'force_c',
            'lift_height', 'state'
        ])
    
    def _angle_from_accel_roll(self, ax, ay, az) -> float:
        """Compute roll angle from accelerometer data (degrees)."""
        return np.degrees(np.arctan2(ay, np.sqrt(ax**2 + az**2)))
    
    def _angle_from_accel_pitch(self, ax, ay, az) -> float:
        """Compute pitch angle from accelerometer data (degrees)."""
        return np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
    
    def _read_imu(self) -> dict:
        """
        Read IMU data from ESP32 interface.
        
        Returns:
            Dict with roll, pitch, yaw, and gyro data (in radians)
        """
        if self.mode == Mode.SIMULATION:
            # Simulate small disturbances
            return {
                'roll': np.random.normal(0, 0.01),
                'pitch': np.random.normal(0, 0.01),
                'yaw': 0.0,
                'gyro_x': 0.0,
                'gyro_y': 0.0,
                'gyro_z': 0.0,
            }
        
        if self.esp32 is not None:
            try:
                # Read from ESP32's integrated IMU
                imu_data = self.esp32.imu_get_data()
                
                # Compute roll/pitch from accelerometer
                roll_deg = self._angle_from_accel_roll(
                    imu_data['ax'], imu_data['ay'], imu_data['az']
                )
                pitch_deg = self._angle_from_accel_pitch(
                    imu_data['ax'], imu_data['ay'], imu_data['az']
                )
                
                return {
                    'roll': np.radians(roll_deg),
                    'pitch': np.radians(pitch_deg),
                    'yaw': 0.0,  # Yaw requires magnetometer or integration
                    'gyro_x': imu_data.get('gx', 0),
                    'gyro_y': imu_data.get('gy', 0),
                    'gyro_z': imu_data.get('gz', 0),
                }
            except Exception as e:
                print(f"[LQR Balancer] IMU read error: {e}")
        
        return {
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
        }
    
    def _read_joint_angles(self) -> dict:
        """
        Read current joint angles.
        
        Returns:
            Dict mapping joint names to angles
        """
        # In simulation or without direct feedback, estimate from commands
        joint_angles = {}
        
        for i in range(4):
            # Approximate from stance
            # These would come from encoders in a real system
            joint_angles[f'leg{i}_abd'] = 0.0
            joint_angles[f'leg{i}_hip'] = 0.0
            joint_angles[f'leg{i}_knee'] = -0.5  # Nominal bent position
        
        # Add convenience accessors
        joint_angles['front_roll'] = joint_angles.get('leg0_abd', 0)
        joint_angles['front_pitch'] = joint_angles.get('leg0_hip', 0)
        joint_angles['hind_roll'] = joint_angles.get('leg3_abd', 0)
        joint_angles['hind_pitch'] = joint_angles.get('leg3_hip', 0)
        
        return joint_angles
    
    def _compute_stance(self, lift_progress: float) -> np.ndarray:
        """
        Compute stance with lifted legs.
        
        Args:
            lift_progress: 0 to 1, how much legs are lifted
            
        Returns:
            3x4 stance matrix
        """
        stance = DEFAULT_STANCE_MATRIX.copy()
        
        # Lift the designated legs
        lift_amount = lift_progress * self.target_lift_height
        
        for leg_idx in self.lift_legs:
            stance[2, leg_idx] += lift_amount  # Positive Z is up
        
        return stance
    
    def _apply_balance_correction(self, stance: np.ndarray, 
                                   control_output: dict,
                                   imu_data: dict) -> np.ndarray:
        """
        Apply simple balance correction to stance with rate limiting.
        
        Uses direct IMU feedback to shift support foot positions.
        If robot tilts right (positive roll), shift feet left to compensate.
        If robot tilts forward (positive pitch), shift feet back to compensate.
        
        Args:
            stance: Current stance matrix
            control_output: Output from 6-DOF controller (for logging)
            imu_data: Current IMU readings
            
        Returns:
            Modified stance matrix
        """
        corrected_stance = stance.copy()
        
        # Get current tilt (already baseline-compensated)
        roll = imu_data.get('roll', 0)   # Radians
        pitch = imu_data.get('pitch', 0)  # Radians
        
        # Calculate desired corrections
        desired_y = -roll * self.balance_gain_y   # Opposite direction to tilt
        desired_x = -pitch * self.balance_gain_x  # Opposite direction to tilt
        
        # Rate limit the corrections for smooth movement
        max_delta = self.max_correction_rate * CONTROL_DT
        
        # Smoothly move toward desired correction
        x_delta = desired_x - self.last_x_correction
        y_delta = desired_y - self.last_y_correction
        
        # Clamp the rate of change
        x_delta = np.clip(x_delta, -max_delta, max_delta)
        y_delta = np.clip(y_delta, -max_delta, max_delta)
        
        # Update corrections
        x_correction = self.last_x_correction + x_delta
        y_correction = self.last_y_correction + y_delta
        
        # Store for next iteration
        self.last_x_correction = x_correction
        self.last_y_correction = y_correction
        
        # Apply to support legs only
        for leg_idx in self.support_legs:
            corrected_stance[0, leg_idx] += x_correction
            corrected_stance[1, leg_idx] += y_correction
        
        return corrected_stance
    
    def _command_stance(self, stance: np.ndarray):
        """
        Command the robot to a given stance.
        
        Args:
            stance: 3x4 stance matrix
        """
        if self.mode == Mode.SIMULATION:
            return
        
        if self.hardware is None or self.config_obj is None:
            return
        
        try:
            # Compute inverse kinematics
            joint_angles = self._four_legs_ik(stance, self.config_obj)
            
            # Send to hardware (note: method has typo in original API)
            self.hardware.set_actuator_postions(joint_angles)
            
        except Exception as e:
            print(f"[LQR Balancer] Command error: {e}")
    
    def _check_safety(self, state: dict) -> bool:
        """
        Check if robot is within safety limits.
        
        Args:
            state: Estimated state from BalanceStateEstimator
            
        Returns:
            True if safe, False if emergency stop needed
        """
        # Check tilt angles (relative to baseline, not absolute)
        theta_k = abs(state.get('theta_k', 0))
        theta_perp = abs(state.get('theta_perp', 0))
        
        total_tilt = np.sqrt(theta_k**2 + theta_perp**2)
        
        if total_tilt > SafetyLimits.MAX_BODY_TILT:
            print(f"[LQR Balancer] SAFETY: Tilt {np.rad2deg(total_tilt):.1f}° exceeds limit!")
            return False
        
        # Check stability margin - use configurable threshold
        min_margin = getattr(SafetyLimits, 'MIN_STABILITY_MARGIN', 0.05)
        if state.get('stability_margin', 1.0) < min_margin:
            print(f"[LQR Balancer] SAFETY: Stability margin {state.get('stability_margin', 0):.2f} < {min_margin}")
            return False
        
        return True
    
    def calibrate_baseline(self, samples: int = 30):
        """
        Calibrate IMU baseline to compensate for resting position.
        
        Args:
            samples: Number of samples to average
        """
        print("[LQR Balancer] Calibrating IMU baseline...")
        
        roll_sum = 0.0
        pitch_sum = 0.0
        valid_samples = 0
        
        for i in range(samples):
            imu_data = self._read_imu()
            if imu_data['roll'] != 0.0 or imu_data['pitch'] != 0.0:
                roll_sum += imu_data['roll']
                pitch_sum += imu_data['pitch']
                valid_samples += 1
            time.sleep(0.02)
        
        if valid_samples > 0:
            self.baseline_roll = roll_sum / valid_samples
            self.baseline_pitch = pitch_sum / valid_samples
        
        print(f"[LQR Balancer] Baseline: roll={np.rad2deg(self.baseline_roll):.1f}°, "
              f"pitch={np.rad2deg(self.baseline_pitch):.1f}°")
    
    def _read_imu_calibrated(self) -> dict:
        """Read IMU data with baseline compensation."""
        imu_data = self._read_imu()
        
        # Subtract baseline
        imu_data['roll'] -= self.baseline_roll
        imu_data['pitch'] -= self.baseline_pitch
        
        return imu_data
    
    def _log_state(self, time_elapsed: float, imu_data: dict, 
                   state: dict, control_output: dict, phase: str):
        """Log data for analysis."""
        if not LOG_TO_FILE or self.log_writer is None:
            return
        
        self.log_writer.writerow([
            f"{time_elapsed:.3f}",
            f"{imu_data.get('roll', 0):.4f}",
            f"{imu_data.get('pitch', 0):.4f}",
            f"{imu_data.get('yaw', 0):.4f}",
            f"{state.get('theta_k', 0):.4f}",
            f"{state.get('theta_perp', 0):.4f}",
            f"{state.get('body_height', 0):.4f}",
            f"{state.get('stability_margin', 0):.3f}",
            f"{control_output['part_a_torques'][0]:.4f}",
            f"{control_output['part_a_torques'][1]:.4f}",
            f"{control_output['part_b_torque']:.4f}",
            f"{control_output['part_c_force']:.4f}",
            f"{self.lift_height:.4f}",
            phase
        ])
    
    def run(self, hold_time: float = None):
        """
        Run the diagonal balance experiment.
        
        Args:
            hold_time: How long to hold the diagonal stance (default from config)
        """
        if hold_time is None:
            hold_time = HOLD_TIME
        
        # Initialize
        if not self.initialize_hardware():
            print("[LQR Balancer] Cannot run without hardware")
            return
        
        self.initialize_logging()
        self.controller.reset()
        self.running = True
        
        print("\n" + "="*60)
        print("LQR DIAGONAL BALANCE EXPERIMENT")
        print("="*60)
        print(f"Support pair: {self.support_pair}")
        print(f"Support legs: {self.support_legs}")
        print(f"Lift legs: {self.lift_legs}")
        print(f"Target lift: {self.target_lift_height*1000:.0f} mm")
        print(f"Hold time: {hold_time:.1f} s")
        print("="*60 + "\n")
        
        # Print LQR stability info
        stability_info = self.controller.get_diagnostics()
        lqr_info = stability_info['part_a_stability']
        print(f"[LQR] Closed-loop stable: {lqr_info['stable']}")
        if lqr_info['stable']:
            print(f"[LQR] Approx settling time: {lqr_info['settling_time_approx']:.2f} s")
        
        # Calibrate IMU baseline before starting
        self.calibrate_baseline()
        
        self.start_time = time.time()
        phase = "LIFT"
        
        try:
            while self.running and not self.emergency_stop:
                loop_start = time.time()
                time_elapsed = loop_start - self.start_time
                
                # Determine current phase
                if time_elapsed < LIFT_TIME:
                    phase = "LIFT"
                    lift_progress = time_elapsed / LIFT_TIME
                elif time_elapsed < LIFT_TIME + hold_time:
                    phase = "HOLD"
                    lift_progress = 1.0
                elif time_elapsed < LIFT_TIME + hold_time + RETURN_TIME:
                    phase = "RETURN"
                    return_elapsed = time_elapsed - LIFT_TIME - hold_time
                    lift_progress = 1.0 - (return_elapsed / RETURN_TIME)
                else:
                    phase = "DONE"
                    self.running = False
                    continue
                
                # Read sensors (with baseline compensation)
                imu_data = self._read_imu_calibrated()
                joint_angles = self._read_joint_angles()
                
                # Estimate state
                state = self.state_estimator.estimate_state(
                    imu_data, joint_angles, self.support_legs
                )
                
                # Check safety (only during LIFT and HOLD phases)
                if phase in ["LIFT", "HOLD"] and not self._check_safety(state):
                    print("[LQR Balancer] Emergency stop triggered!")
                    self.emergency_stop = True
                    break
                
                # Compute control
                control_output = self.controller.update(
                    imu_data, joint_angles,
                    state['body_height'],
                    CONTROL_DT
                )
                
                # Compute and apply stance
                self.lift_height = lift_progress * self.target_lift_height
                base_stance = self._compute_stance(lift_progress)
                
                # Apply balance corrections during LIFT and HOLD phases if enabled
                if phase in ["LIFT", "HOLD"] and self.enable_active_balance:
                    corrected_stance = self._apply_balance_correction(
                        base_stance, control_output, imu_data
                    )
                else:
                    corrected_stance = base_stance
                
                # Command robot
                self._command_stance(corrected_stance)
                
                # Log data
                self._log_state(time_elapsed, imu_data, state, control_output, phase)
                
                # Debug output
                if DEBUG_PRINT and int(time_elapsed * 10) % 5 == 0:
                    self._print_status(time_elapsed, phase, state, control_output)
                
                # Maintain loop timing
                loop_duration = time.time() - loop_start
                sleep_time = CONTROL_DT - loop_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            print(f"[LQR Balancer] Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.stop()
    
    def _print_status(self, time_elapsed: float, phase: str, 
                      state: dict, control_output: dict):
        """Print status update."""
        print(f"\r[{phase:6s}] t={time_elapsed:5.1f}s | "
              f"θk={np.rad2deg(state['theta_k']):+5.1f}° | "
              f"θ⊥={np.rad2deg(state['theta_perp']):+5.1f}° | "
              f"τA=[{control_output['part_a_torques'][0]:+.2f},{control_output['part_a_torques'][1]:+.2f}] | "
              f"S={state['stability_margin']:.2f}", 
              end='', flush=True)
    
    def stop(self):
        """Stop the balancer and return to safe stance."""
        print("\n[LQR Balancer] Stopping...")
        self.running = False
        
        # Return to default stance
        if self.mode != Mode.SIMULATION:
            print("[LQR Balancer] Returning to default stance...")
            self._command_stance(DEFAULT_STANCE_MATRIX)
            time.sleep(0.5)
        
        # Close log file
        if self.log_file is not None:
            self.log_file.close()
            print(f"[LQR Balancer] Log saved to {LOG_FILE}")
        
        # Deactivate hardware
        if self.hardware is not None:
            try:
                self.hardware.deactivate()
            except:
                pass
        
        print("[LQR Balancer] Stopped")
    
    def run_simulation(self, duration: float = 5.0, 
                       disturbance: dict = None) -> list:
        """
        Run a pure simulation (no hardware).
        
        Args:
            duration: Simulation duration in seconds
            disturbance: Optional disturbance to apply
            
        Returns:
            List of logged states for analysis
        """
        print("\n[LQR Balancer] Running simulation...")
        
        self.controller.reset()
        
        # Simulation state
        state_history = []
        current_body_state = np.zeros(6)  # [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
        
        # Apply initial disturbance if specified
        if disturbance is not None:
            if 'roll_velocity' in disturbance:
                current_body_state[5] = disturbance['roll_velocity']
            if 'pitch_velocity' in disturbance:
                current_body_state[5] += disturbance['pitch_velocity']
        
        t = 0.0
        while t < duration:
            # Convert state to IMU-like readings
            imu_data = {
                'roll': current_body_state[4] * 0.5,  # Approximate
                'pitch': current_body_state[4] * 0.5,
                'yaw': 0.0,
                'gyro_x': current_body_state[5] * 0.5,
                'gyro_y': current_body_state[5] * 0.5,
                'gyro_z': 0.0,
            }
            
            joint_angles = {
                'front_roll': current_body_state[0] * 0.1,
                'front_pitch': current_body_state[0],
                'hind_roll': current_body_state[2] * 0.1,
                'hind_pitch': current_body_state[2],
            }
            
            # Compute control
            control_output = self.controller.update(
                imu_data, joint_angles,
                RobotParams.STANDING_HEIGHT,
                CONTROL_DT
            )
            
            # Simulate dynamics (simplified)
            # Apply control torques through the dynamics model
            tau = control_output['part_a_torques']
            
            # Simple integration (the full dynamics are in the Part A controller)
            from experiments.lqr_diagonal_balance.dynamics import DiagonalDynamics
            dynamics = DiagonalDynamics()
            current_body_state = dynamics.simulate_step(
                current_body_state, tau, CONTROL_DT
            )
            
            # Record
            state_history.append({
                'time': t,
                'state': current_body_state.copy(),
                'control': tau.copy(),
                'theta_body': current_body_state[4],
            })
            
            t += CONTROL_DT
        
        # Print summary
        final_theta = state_history[-1]['theta_body']
        max_theta = max(abs(s['theta_body']) for s in state_history)
        
        print(f"[Simulation] Final body angle: {np.rad2deg(final_theta):.2f}°")
        print(f"[Simulation] Max body angle: {np.rad2deg(max_theta):.2f}°")
        print(f"[Simulation] Stable: {abs(final_theta) < 0.05}")
        
        return state_history
