"""
Main Controller for Reaction Wheel LQR Balance.

Implementation of the control loop:
1. Read IMU + encoders
2. Compute θ_rel, θ_body (with drift compensation)
3. Form state vector X
4. Compute control torque u = -k^T X
5. Apply cooperative correction u_c
6. Send u_front, u_rear to hip actuators
7. Clamp torques within safe limits
"""

import numpy as np
import time
import signal
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.new_lqr_balance.config import (
    RobotParams, ControlParams, SafetyParams, SensorParams,
    CONTROL_DT, TIMEOUT_S, SUPPORT_PAIR, DIAGONAL_PAIRS,
    DEFAULT_STANCE, DEBUG_PRINT, DEBUG_INTERVAL, LOG_TO_FILE, LOG_DIR,
    LIFT_TIME, HOLD_TIME, RETURN_TIME
)
from experiments.new_lqr_balance.state_estimator import StateEstimator, EncoderSimulator
from experiments.new_lqr_balance.lqr_controller import ReactionWheelLQR
from experiments.new_lqr_balance.torque_converter import TorqueToPositionConverter
from experiments.new_lqr_balance.safety import SafetyMonitor, TorqueLimiter, DataLogger


class OperatingMode:
    """Operating modes for the controller."""
    SIMULATION = "simulation"
    BENCH = "bench"        # On bench with legs free
    PHYSICAL = "physical"  # On ground


class ReactionWheelBalancer:
    """
    Main controller for reaction wheel LQR balance.
    
    Implements the complete control loop from the spec:
    1. Read IMU + encoders
    2. Compute θ_rel, θ_body (with drift compensation)
    3. Form state vector X
    4. Compute control torque u = -k^T X
    5. Apply cooperative correction u_c
    6. Send u_front, u_rear to hip actuators
    7. Clamp torques within safe limits
    """
    
    def __init__(self, mode: str = OperatingMode.SIMULATION):
        """
        Initialize balancer.
        
        Args:
            mode: Operating mode (simulation, bench, physical)
        """
        self.mode = mode
        self.running = False
        
        # Get diagonal configuration
        self.support_pair = SUPPORT_PAIR
        self.diagonal_config = DIAGONAL_PAIRS[self.support_pair]
        self.support_legs = self.diagonal_config["support"]
        self.lift_legs = self.diagonal_config["lift"]
        
        # Components
        self.state_estimator = StateEstimator(dt=CONTROL_DT)
        self.lqr_controller = ReactionWheelLQR()
        self.torque_converter = TorqueToPositionConverter()
        self.safety_monitor = SafetyMonitor()
        self.torque_limiter = TorqueLimiter()
        self.encoder_sim = EncoderSimulator()
        
        # Logging
        self.logger = DataLogger() if LOG_TO_FILE else None
        
        # State
        self.current_stance = DEFAULT_STANCE.copy()
        self.current_joint_angles = None  # Will be initialized by hardware
        self.lift_height = 0.0
        self.target_lift_height = 0.02  # 2cm lift
        
        # Timing
        self.start_time = None
        self.last_loop_time = None
        self.last_debug_time = 0.0
        self.loop_count = 0
        
        # Hardware interfaces (initialized if needed)
        self.hardware = None
        self.esp32 = None
        self.config_obj = None
        self._four_legs_ik = None
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n[Balancer] Received interrupt, stopping...")
        self.emergency_stop = True
        self.stop()
    
    def initialize_hardware(self) -> bool:
        """
        Initialize hardware interfaces.
        
        Returns:
            True if successful
        """
        if self.mode == OperatingMode.SIMULATION:
            print("[Balancer] Simulation mode - no hardware")
            return True
        
        try:
            print("[Balancer] Initializing hardware...")
            
            # Import hardware modules
            from MangDang.mini_pupper.HardwareInterface import HardwareInterface
            from MangDang.mini_pupper.Config import Configuration
            from MangDang.mini_pupper.ESP32Interface import ESP32Interface
            from pupper.Kinematics import four_legs_inverse_kinematics
            
            # Store kinematics function
            self._four_legs_ik = four_legs_inverse_kinematics
            
            # Create instances
            self.config_obj = Configuration()
            self.hardware = HardwareInterface()
            self.esp32 = ESP32Interface()
            
            print("[Balancer] Hardware initialized successfully")
            return True
            
        except Exception as e:
            print(f"[Balancer] Hardware initialization failed: {e}")
            return False
    
    def _read_imu(self) -> dict:
        """
        Read IMU data.
        
        Returns:
            Dict with roll, pitch, gyro_x, gyro_y, gyro_z
        """
        if self.mode == OperatingMode.SIMULATION:
            # Simulate with small noise
            return {
                'roll': np.random.normal(0, 0.005),
                'pitch': np.random.normal(0, 0.005),
                'yaw': 0.0,
                'gyro_x': np.random.normal(0, 0.01),
                'gyro_y': np.random.normal(0, 0.01),
                'gyro_z': 0.0,
            }
        
        if self.esp32 is not None:
            try:
                imu_data = self.esp32.imu_get_data()
                
                # Compute angles from accelerometer
                ax = imu_data.get('ax', 0)
                ay = imu_data.get('ay', 0)
                az = imu_data.get('az', -1)
                
                roll = np.arctan2(ay, np.sqrt(ax**2 + az**2))
                pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
                
                return {
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': imu_data.get('yaw', 0),
                    'gyro_x': imu_data.get('gx', 0),
                    'gyro_y': imu_data.get('gy', 0),
                    'gyro_z': imu_data.get('gz', 0),
                }
            except Exception as e:
                print(f"[Balancer] IMU read error: {e}")
        
        return {'roll': 0, 'pitch': 0, 'yaw': 0, 'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0}
    
    def _read_encoders(self) -> dict:
        """
        Read encoder data (or simulate from commanded positions).
        
        Returns:
            Dict with theta_front, theta_rear
        """
        if self.current_joint_angles is not None:
            return self.encoder_sim.update_from_commanded(
                self.current_joint_angles, self.support_legs
            )
        
        return {'theta_front': 0.0, 'theta_rear': 0.0}
    
    def _compute_stance_with_lift(self, lift_progress: float) -> np.ndarray:
        """
        Compute stance with lifted legs.
        
        Args:
            lift_progress: 0-1 progress of leg lift
            
        Returns:
            3x4 stance matrix
        """
        stance = DEFAULT_STANCE.copy()
        
        # Lift designated legs
        lift_amount = lift_progress * self.target_lift_height
        
        for leg_idx in self.lift_legs:
            stance[2, leg_idx] += lift_amount  # Positive Z is up
        
        return stance
    
    def _command_stance(self, stance: np.ndarray):
        """
        Command robot to stance position.
        
        Args:
            stance: 3x4 stance matrix
        """
        if self.mode == OperatingMode.SIMULATION:
            return
        
        if self.hardware is None or self.config_obj is None:
            return
        
        try:
            # Compute inverse kinematics
            joint_angles = self._four_legs_ik(stance, self.config_obj)
            self.current_joint_angles = joint_angles
            
            # Send to hardware
            self.hardware.set_actuator_postions(joint_angles)
            
        except Exception as e:
            print(f"[Balancer] Command error: {e}")
    
    def _command_joint_angles(self, joint_angles: np.ndarray):
        """
        Command joint angles directly.
        
        Args:
            joint_angles: 3x4 joint angle matrix
        """
        if self.mode == OperatingMode.SIMULATION:
            self.current_joint_angles = joint_angles
            return
        
        if self.hardware is None:
            return
        
        try:
            self.hardware.set_actuator_postions(joint_angles)
            self.current_joint_angles = joint_angles
        except Exception as e:
            print(f"[Balancer] Command error: {e}")
    
    def calibrate(self, samples: int = 30):
        """
        Calibrate IMU baseline.
        
        Args:
            samples: Number of samples to average
        """
        print("[Balancer] Calibrating IMU baseline...")
        
        roll_sum = 0.0
        pitch_sum = 0.0
        count = 0
        
        for _ in range(samples):
            imu_data = self._read_imu()
            roll_sum += imu_data['roll']
            pitch_sum += imu_data['pitch']
            count += 1
            time.sleep(0.02)
        
        if count > 0:
            baseline = {
                'roll': roll_sum / count,
                'pitch': pitch_sum / count,
            }
            self.state_estimator.calibrate(baseline)
            print(f"[Balancer] Baseline: roll={np.degrees(baseline['roll']):.1f}°, "
                  f"pitch={np.degrees(baseline['pitch']):.1f}°")
    
    def control_loop_step(self) -> dict:
        """
        Execute one step of the control loop.
        
        Returns:
            Dict with loop data for debugging
        """
        # === 1. Read sensors ===
        imu_data = self._read_imu()
        encoder_data = self._read_encoders()
        
        # === 2. Estimate state ===
        state = self.state_estimator.update(imu_data, encoder_data)
        
        # === 3. Check safety ===
        safety_status = self.safety_monitor.check_state(state)
        
        if not safety_status['safe']:
            if safety_status.get('fall_detected'):
                print(f"[Balancer] FALL DETECTED: {safety_status['message']}")
                self.emergency_stop = True
                return {'emergency_stop': True, 'reason': safety_status['message']}
            elif safety_status.get('action') == 'reduce_gain':
                # Could implement gain scheduling here
                pass
        
        # === 4. Compute LQR control ===
        base_torque = self.lqr_controller.compute_control(state)
        
        # === 5. Apply cooperative distribution ===
        u_front, u_rear = self.lqr_controller.distribute_torque(
            base_torque, encoder_data, state[0]  # state[0] = theta_rel
        )
        
        # === 6. Apply torque limits ===
        u_front, sat_front = self.torque_limiter.limit(u_front)
        u_rear, sat_rear = self.torque_limiter.limit(u_rear)
        
        # === 7. Convert torque to position commands ===
        position_offsets = self.torque_converter.convert(u_front, u_rear)
        
        # === 8. Apply to joint angles ===
        if self.current_joint_angles is not None:
            modified_angles = self.torque_converter.apply_to_joint_angles(
                self.current_joint_angles, position_offsets, self.support_legs
            )
            self._command_joint_angles(modified_angles)
        
        # === 9. Log data ===
        if self.logger:
            self.logger.log_state(state, imu_data, encoder_data)
            self.logger.log_control(base_torque, u_front, u_rear, position_offsets)
            if not safety_status['safe']:
                self.logger.log_safety(safety_status)
        
        # === 10. Debug output ===
        self.loop_count += 1
        current_time = time.time()
        
        if DEBUG_PRINT and (current_time - self.last_debug_time) > DEBUG_INTERVAL:
            state_info = self.state_estimator.get_detailed_state()
            print(f"[{self.loop_count:5d}] θ_body={state_info['theta_body_deg']:6.2f}° "
                  f"θ_rel={state_info['theta_rel_deg']:6.2f}° "
                  f"τ_base={base_torque:6.3f} "
                  f"τ_f={u_front:6.3f} τ_r={u_rear:6.3f}")
            self.last_debug_time = current_time
        
        return {
            'state': state,
            'base_torque': base_torque,
            'u_front': u_front,
            'u_rear': u_rear,
            'position_offsets': position_offsets,
            'safety_status': safety_status,
        }
    
    def run_phase_lift_legs(self, duration: float = 2.0):
        """
        Phase 1: Gradually lift the non-support legs.
        
        Args:
            duration: Time to complete lift
        """
        print("[Balancer] Phase 1: Lifting legs...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.emergency_stop:
                break
            
            progress = (time.time() - start_time) / duration
            progress = min(progress, 1.0)
            
            stance = self._compute_stance_with_lift(progress)
            self._command_stance(stance)
            
            time.sleep(CONTROL_DT)
        
        self.lift_height = self.target_lift_height
        print(f"[Balancer] Legs lifted to {self.lift_height*100:.1f}cm")
    
    def run_phase_balance(self, duration: float = 30.0):
        """
        Phase 2: Active balance control.
        
        Args:
            duration: How long to balance
        """
        print("[Balancer] Phase 2: Active balance control...")
        
        # Initialize joint angles from stance
        stance = self._compute_stance_with_lift(1.0)
        self._command_stance(stance)
        time.sleep(0.1)  # Let servos settle
        
        # If we have IK, get initial joint angles
        if self._four_legs_ik is not None and self.config_obj is not None:
            self.current_joint_angles = self._four_legs_ik(stance, self.config_obj)
        else:
            # Create default joint angles
            self.current_joint_angles = np.zeros((3, 4))
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.emergency_stop:
                break
            
            loop_start = time.time()
            
            # Execute control step
            result = self.control_loop_step()
            
            if result.get('emergency_stop'):
                break
            
            # Maintain loop timing
            elapsed = time.time() - loop_start
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print(f"[Balancer] Balance phase complete. Loops: {self.loop_count}")
    
    def run_phase_lower_legs(self, duration: float = 2.0):
        """
        Phase 3: Lower legs back to ground.
        
        Args:
            duration: Time to complete lowering
        """
        print("[Balancer] Phase 3: Lowering legs...")
        
        start_time = time.time()
        initial_lift = self.lift_height
        
        while time.time() - start_time < duration:
            progress = (time.time() - start_time) / duration
            progress = min(progress, 1.0)
            
            # Interpolate from current lift to zero
            current_lift = initial_lift * (1.0 - progress)
            self.lift_height = current_lift
            
            stance = self._compute_stance_with_lift(current_lift / self.target_lift_height)
            self._command_stance(stance)
            
            time.sleep(CONTROL_DT)
        
        print("[Balancer] Legs lowered")
    
    def run(self, hold_time: float = None):
        """
        Run the complete balance sequence with integrated phase control.
        
        Uses phase-based control like lqr_diagonal_balance:
        - LIFT: Gradually lift legs over LIFT_TIME
        - HOLD: Active balance control for hold_time
        - RETURN: Lower legs over RETURN_TIME
        
        Args:
            hold_time: How long to balance (uses HOLD_TIME from config if None)
        """
        if hold_time is None:
            hold_time = HOLD_TIME
        
        print("\n" + "=" * 60)
        print("REACTION WHEEL LQR BALANCE CONTROLLER")
        print("=" * 60)
        print(f"Mode: {self.mode}")
        print(f"Support pair: {self.support_pair}")
        print(f"Support legs: {self.support_legs}")
        print(f"Lift legs: {self.lift_legs}")
        print(f"Target lift: {self.target_lift_height*1000:.0f} mm")
        print(f"Timing: lift={LIFT_TIME:.1f}s, hold={hold_time:.1f}s, return={RETURN_TIME:.1f}s")
        print("=" * 60 + "\n")
        
        # Initialize
        if not self.initialize_hardware():
            print("[Balancer] Failed to initialize hardware")
            return
        
        # Move to initial crouched stance
        print("[Balancer] Moving to initial stance...")
        self._command_stance(DEFAULT_STANCE)
        time.sleep(1.0)
        
        # Initialize joint angles from stance
        if self._four_legs_ik is not None and self.config_obj is not None:
            self.current_joint_angles = self._four_legs_ik(DEFAULT_STANCE, self.config_obj)
        else:
            self.current_joint_angles = np.zeros((3, 4))
        
        # Calibrate IMU
        self.calibrate()
        
        self.start_time = time.time()
        self.running = True
        phase = "LIFT"
        
        total_time = LIFT_TIME + hold_time + RETURN_TIME
        
        try:
            while self.running and not self.emergency_stop:
                loop_start = time.time()
                time_elapsed = loop_start - self.start_time
                
                # Determine current phase and lift progress
                if time_elapsed < LIFT_TIME:
                    phase = "LIFT"
                    lift_progress = time_elapsed / LIFT_TIME
                elif time_elapsed < LIFT_TIME + hold_time:
                    phase = "HOLD"
                    lift_progress = 1.0
                elif time_elapsed < total_time:
                    phase = "RETURN"
                    return_elapsed = time_elapsed - LIFT_TIME - hold_time
                    lift_progress = 1.0 - (return_elapsed / RETURN_TIME)
                else:
                    phase = "DONE"
                    self.running = False
                    continue
                
                # Compute stance with current lift
                self.lift_height = lift_progress * self.target_lift_height
                stance = self._compute_stance_with_lift(lift_progress)
                
                # Read sensors
                imu_data = self._read_imu()
                encoder_data = self._read_encoders()
                
                # Update state estimate
                state = self.state_estimator.update(imu_data, encoder_data)
                
                # Apply active balance control during LIFT and HOLD phases
                if phase in ["LIFT", "HOLD"]:
                    # Check safety
                    safety_status = self.safety_monitor.check_state(state)
                    if not safety_status['safe']:
                        if safety_status.get('fall_detected'):
                            print(f"[Balancer] FALL DETECTED: {safety_status['message']}")
                            self.emergency_stop = True
                            break
                    
                    # Compute LQR control
                    base_torque = self.lqr_controller.compute_control(state)
                    
                    # Distribute torque cooperatively
                    u_front, u_rear = self.lqr_controller.distribute_torque(
                        base_torque, encoder_data, state[0]
                    )
                    
                    # Apply torque limits
                    u_front, _ = self.torque_limiter.limit(u_front)
                    u_rear, _ = self.torque_limiter.limit(u_rear)
                    
                    # Convert torque to position adjustments
                    position_offsets = self.torque_converter.convert(u_front, u_rear)
                    
                    # Apply balance corrections to joint angles
                    if self.current_joint_angles is not None:
                        modified_angles = self.torque_converter.apply_to_joint_angles(
                            self.current_joint_angles, position_offsets, self.support_legs
                        )
                        self._command_joint_angles(modified_angles)
                    else:
                        self._command_stance(stance)
                    
                    # Log control data
                    if self.logger:
                        self.logger.log_state(state, imu_data, encoder_data)
                        self.logger.log_control(base_torque, u_front, u_rear, position_offsets)
                else:
                    # RETURN phase - just command stance, no active balance
                    self._command_stance(stance)
                    base_torque = 0.0
                    u_front = u_rear = 0.0
                
                # Debug output
                self.loop_count += 1
                current_time = time.time()
                
                if DEBUG_PRINT and (current_time - self.last_debug_time) > DEBUG_INTERVAL:
                    state_info = self.state_estimator.get_detailed_state()
                    print(f"[{phase:6s}] t={time_elapsed:5.1f}s "
                          f"lift={self.lift_height*100:4.1f}cm "
                          f"θ_body={state_info['theta_body_deg']:6.2f}° "
                          f"τ={base_torque:6.3f}")
                    self.last_debug_time = current_time
                
                # Maintain loop timing
                loop_duration = time.time() - loop_start
                sleep_time = CONTROL_DT - loop_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print(f"\n[Balancer] Sequence complete. Total loops: {self.loop_count}")
            
        except Exception as e:
            print(f"[Balancer] Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the controller and return to safe state."""
        print("[Balancer] Stopping...")
        self.running = False
        
        # Return to neutral stance
        if self.hardware is not None:
            try:
                self._command_stance(DEFAULT_STANCE)
            except:
                pass
        
        # Save log
        if self.logger:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(LOG_DIR, exist_ok=True)
            filename = os.path.join(LOG_DIR, f"balance_log_{timestamp}.csv")
            self.logger.save_to_file(filename)
            
            summary = self.logger.get_summary()
            print("\n[Balancer] Session Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        
        print("[Balancer] Stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reaction Wheel LQR Balance")
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['simulation', 'bench', 'physical'],
                        help='Operating mode')
    parser.add_argument('--hold-time', type=float, default=None,
                        help='Balance hold duration in seconds (default: from config)')
    parser.add_argument('--support', type=str, default='FL_BR',
                        choices=['FL_BR', 'FR_BL'],
                        help='Which diagonal pair to use for support')
    
    args = parser.parse_args()
    
    # Create and run balancer
    balancer = ReactionWheelBalancer(mode=args.mode)
    balancer.support_pair = args.support
    balancer.diagonal_config = DIAGONAL_PAIRS[args.support]
    balancer.support_legs = balancer.diagonal_config["support"]
    balancer.lift_legs = balancer.diagonal_config["lift"]
    
    balancer.run(hold_time=args.hold_time)


if __name__ == "__main__":
    main()
