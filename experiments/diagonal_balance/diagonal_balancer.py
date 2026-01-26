"""
Diagonal Balancer - Main Control Loop for Mini Pupper 2.

This is the main entry point for the diagonal balance experiment.
It orchestrates:
- IMU reading and sensor fusion
- Safety state machine
- Movement generation and execution
- PID balance correction

Usage:
    python -m experiments.diagonal_balance.diagonal_balancer --mode bench
    python -m experiments.diagonal_balance.diagonal_balancer --mode physical
"""

import argparse
import numpy as np
import time
import sys
import os
import signal

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.diagonal_balance import config
from experiments.diagonal_balance.diagonal_movement import DiagonalMovementGenerator
from experiments.diagonal_balance.filters import (
    SensorFusion, PIDController, 
    angle_from_accel_roll, angle_from_accel_pitch
)
from experiments.diagonal_balance.safety import (
    SafetyManager, BalanceState, EmergencyStop
)


class DiagonalBalancer:
    """
    Main controller for diagonal leg balancing.
    
    Integrates all components and runs the control loop.
    """
    
    def __init__(self, mode="bench"):
        """
        Initialize the diagonal balancer.
        
        Args:
            mode: Operating mode - "bench" or "physical"
        """
        self.mode = mode
        self.running = False
        
        # Movement generator
        self.movement_gen = DiagonalMovementGenerator()
        
        # Sensor fusion
        self.sensor_fusion = SensorFusion()
        
        # PID controllers for roll and pitch
        self.pid_roll = PIDController(setpoint=0.0)
        self.pid_pitch = PIDController(setpoint=0.0)
        
        # Safety manager
        self.safety = SafetyManager()
        
        # Hardware interfaces (initialized later if needed)
        self.hardware = None
        self.controller = None
        self.esp32 = None
        self.imu = None
        self.disp = None
        
        # Emergency stop
        self.emergency_stop = EmergencyStop()
        
        # Movement execution
        self.movement_scheme = None
        self.movement_lib = None
        self.movement_index = 0
        
        # State
        self.original_roll = 0.0
        self.original_pitch = 0.0
        
        # Active balance corrections - foot position shifts (meters)
        self.active_foot_shift_x = 0.0  # Forward/backward shift
        self.active_foot_shift_y = 0.0  # Left/right shift
        
        # Current diagonal pair being used
        self.current_pair = config.DIAGONAL_PAIR
        
        # Timing
        self.last_loop_time = 0.0
        self.previous_time = 0.0
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n[Balancer] Received interrupt signal, stopping...")
        self.stop()
    
    def initialize_hardware(self):
        """Initialize hardware interfaces for physical operation."""
        if self.mode not in ["bench", "physical"]:
            return True
        
        try:
            print("[Balancer] Initializing hardware...")
            
            # Import hardware modules
            from MangDang.mini_pupper.HardwareInterface import HardwareInterface
            from MangDang.mini_pupper.Config import Configuration
            from MangDang.mini_pupper.display import Display
            from MangDang.mini_pupper.ESP32Interface import ESP32Interface
            from pupper.Kinematics import four_legs_inverse_kinematics
            from src.Controller import Controller
            from src.State import State
            from src.MovementScheme import MovementScheme
            from src.Command import Command
            
            # Store imports for later use
            self._MovementScheme = MovementScheme
            self._State = State
            self._Command = Command
            self._four_legs_inverse_kinematics = four_legs_inverse_kinematics
            
            # Create hardware instances
            self.config = Configuration()
            self.hardware = HardwareInterface()
            self.disp = Display()
            self.esp32 = ESP32Interface()
            
            # Create controller
            self.controller = Controller(
                self.config,
                four_legs_inverse_kinematics,
            )
            
            # Create state
            self.state = State()
            # Initialize quaternion orientation (identity quaternion = no rotation)
            self.state.quat_orientation = np.array([1, 0, 0, 0])
            
            # Set up emergency stop with hardware
            self.emergency_stop = EmergencyStop(self.hardware)
            
            # Show IP on display
            self.disp.show_ip()
            
            print("[Balancer] Hardware initialized successfully")
            return True
            
        except Exception as e:
            print(f"[Balancer] Failed to initialize hardware: {e}")
            return False
    
    def calibrate_baseline(self, samples=50):
        """
        Calibrate the baseline roll and pitch angles.
        
        Args:
            samples: Number of samples to average
        """
        if self.esp32 is None:
            self.original_roll = 0.0
            self.original_pitch = 0.0
            return
        
        print("[Balancer] Calibrating baseline angles...")
        
        roll_sum = 0.0
        pitch_sum = 0.0
        
        for i in range(samples):
            try:
                imu_data = self.esp32.imu_get_data()
                roll = angle_from_accel_roll(
                    imu_data['ax'], imu_data['ay'], imu_data['az']
                )
                pitch = angle_from_accel_pitch(
                    imu_data['ax'], imu_data['ay'], imu_data['az']
                )
                roll_sum += roll
                pitch_sum += pitch
                time.sleep(0.01)
            except Exception as e:
                print(f"[Balancer] IMU read error during calibration: {e}")
        
        self.original_roll = roll_sum / samples
        self.original_pitch = pitch_sum / samples
        
        # Set PID setpoints to baseline
        self.pid_roll.set_setpoint(self.original_roll)
        self.pid_pitch.set_setpoint(self.original_pitch)
        
        print(f"[Balancer] Baseline: roll={self.original_roll:.2f}°, pitch={self.original_pitch:.2f}°")
    
    def read_imu(self):
        """
        Read IMU data and return roll/pitch angles.
        
        Returns:
            Tuple of (roll_deg, pitch_deg)
        """
        if self.esp32 is None:
            # Simulate IMU data for testing
            return 0.5 * np.random.randn(), 0.3 * np.random.randn()
        
        try:
            imu_data = self.esp32.imu_get_data()
            roll = angle_from_accel_roll(
                imu_data['ax'], imu_data['ay'], imu_data['az']
            )
            pitch = angle_from_accel_pitch(
                imu_data['ax'], imu_data['ay'], imu_data['az']
            )
            return roll, pitch
        except Exception as e:
            print(f"[Balancer] IMU read error: {e}")
            return 0.0, 0.0
    
    def generate_movements(self, pair=None):
        """
        Generate the diagonal balance movement sequence.
        
        Args:
            pair: Diagonal pair to lift ("FR_BL" or "FL_BR")
        """
        pair = pair if pair is not None else config.DIAGONAL_PAIR
        self.current_pair = pair
        
        print(f"[Balancer] Generating movements for {pair}...")
        
        self.movement_lib = self.movement_gen.generate_full_sequence(pair=pair)
        
        if self.mode in ["bench", "physical"]:
            self.movement_scheme = self._MovementScheme(self.movement_lib)
        
        self.movement_index = 0
        
        print(f"[Balancer] Generated {len(self.movement_lib)} movements")
    
    def _apply_foot_position_correction(self, command):
        """
        Apply balance correction by shifting support leg foot positions.
        
        This modifies command.legslocation directly to shift the feet of
        the support legs, which moves the center of pressure to counteract
        tipping.
        
        Args:
            command: Command object with legslocation to modify
            
        legslocation format: [[x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]]
        - Row 0: X positions for all 4 legs
        - Row 1: Y positions for all 4 legs  
        - Row 2: Z positions for all 4 legs
        - Columns: leg 0=FR, 1=FL, 2=BR, 3=BL
        """
        # Get support leg indices for current pair
        pair_info = config.DIAGONAL_PAIRS.get(self.current_pair)
        if pair_info is None:
            return
        
        support_legs = pair_info["support"]  # e.g., [1, 2] for FL and BR
        
        # Apply shifts to support leg X and Y positions
        for leg_idx in support_legs:
            if leg_idx < 4:  # 4 legs
                # Get current X and Y for this leg
                current_x = command.legslocation[0][leg_idx]  # Row 0 = X
                current_y = command.legslocation[1][leg_idx]  # Row 1 = Y
                
                # Apply shifts
                new_x = current_x + self.active_foot_shift_x
                new_y = current_y + self.active_foot_shift_y
                
                # Clamp to safe limits based on default stance
                base_x = config.DEFAULT_STANCE_MATRIX[0, leg_idx]  # X row
                base_y = config.DEFAULT_STANCE_MATRIX[1, leg_idx]  # Y row
                
                new_x = np.clip(new_x, 
                               base_x - config.MAX_FOOT_SHIFT_X, 
                               base_x + config.MAX_FOOT_SHIFT_X)
                new_y = np.clip(new_y,
                               base_y - config.MAX_FOOT_SHIFT_Y,
                               base_y + config.MAX_FOOT_SHIFT_Y)
                
                # Update the positions
                command.legslocation[0][leg_idx] = new_x
                command.legslocation[1][leg_idx] = new_y
    
    def add_balance_correction(self, error_roll, error_pitch):
        """
        Add a balance correction movement.
        
        Args:
            error_roll: Roll error from PID
            error_pitch: Pitch error from PID
        """
        self.movement_gen.balance_correction(error_roll, error_pitch)
        
        # Update movement scheme with new movement
        if self.mode in ["bench", "physical"] and self.movement_scheme is not None:
            self.movement_lib = self.movement_gen.MovementLib
            self.movement_scheme = self._MovementScheme(self.movement_lib)
    
    def run_movement_step(self):
        """
        Execute one step of the movement scheme.
        
        Returns:
            True if movement step completed, False otherwise
        """
        if self.movement_scheme is None:
            return True
        
        # Run movement scheme
        self.movement_scheme.runMovementScheme()
        
        # Create command
        command = self._Command()
        command.pseudo_dance_event = True
        command.legslocation = self.movement_scheme.getMovemenLegsLocation()
        command.horizontal_velocity = self.movement_scheme.getMovemenSpeed()
        
        # Apply foot position corrections for balance
        # This shifts the support leg feet to move center of pressure
        if self.active_foot_shift_x != 0.0 or self.active_foot_shift_y != 0.0:
            self._apply_foot_position_correction(command)
        
        # Get base attitude from movement scheme (no additional roll/pitch correction)
        command.roll = self.movement_scheme.attitude_now[0]
        command.pitch = self.movement_scheme.attitude_now[1]
        command.yaw = self.movement_scheme.attitude_now[2]
        command.yaw_rate = self.movement_scheme.getMovemenTurn()
        
        # Run controller
        self.controller.run(self.state, command, self.disp)
        
        # Send to hardware
        self.hardware.set_actuator_postions(self.state.joint_angles)
        
        # Check if current movement is complete
        lib_length = len(self.movement_lib)
        current_idx = self.movement_scheme.movement_now_number
        tick = self.movement_scheme.tick
        now_ticks = self.movement_scheme.now_ticks
        
        movement_complete = (current_idx >= lib_length - 1 and tick >= now_ticks)
        
        return movement_complete
    
    def run_bench_test(self):
        """
        Run a single bench test cycle.
        
        Robot should be secured. Executes one complete diagonal lift sequence
        with active balance corrections.
        """
        print("[Balancer] Starting bench test...")
        print("[Balancer] Ensure robot is secured on a stand!")
        print("[Balancer] Press Ctrl+C to abort")
        
        time.sleep(2)  # Give time to read warning
        
        if not self.initialize_hardware():
            print("[Balancer] Hardware initialization failed, aborting")
            return False
        
        # Calibrate baseline
        self.calibrate_baseline()
        
        # Generate movements
        self.generate_movements()
        
        # Start sensor fusion
        self.sensor_fusion.start()
        
        # Run the movement sequence with active balance
        self.running = True
        self.last_loop_time = time.time()
        self.previous_time = time.time()
        
        print("[Balancer] Executing movements with active balance...")
        
        # Track if we're in the lifting/holding phase
        in_balance_phase = False
        balance_start_time = None
        
        try:
            while self.running:
                now = time.time()
                if now - self.last_loop_time < config.CONTROL_LOOP_DT:
                    continue
                self.last_loop_time = now
                
                # Read IMU
                roll, pitch = self.read_imu()
                
                # Filter
                filtered_roll, filtered_pitch = self.sensor_fusion.update(roll, pitch)
                
                # Check safety
                if abs(filtered_roll) > config.MAX_TILT_ABORT or abs(filtered_pitch) > config.MAX_TILT_ABORT:
                    print(f"[Balancer] Tilt abort: roll={filtered_roll:.1f}°, pitch={filtered_pitch:.1f}°")
                    self.emergency_stop.trigger("Excessive tilt")
                    break
                
                # Check current movement phase
                if self.movement_scheme is not None:
                    current_name = self.movement_scheme.movements_now.getMovementName()
                    
                    # Detect when we enter any lift phase (gradual steps or final lift)
                    is_lift_phase = ('gradual_lift' in current_name or 
                                     current_name == 'diagonal_lift')
                    
                    if is_lift_phase and not in_balance_phase:
                        in_balance_phase = True
                        balance_start_time = now
                        print("[Balancer] Entering balance phase (gradual lift)...")
                    
                    # Apply PID balance corrections during any lift phase
                    if in_balance_phase and is_lift_phase:
                        elapsed = now - self.previous_time
                        self.previous_time = now
                        
                        # Compute PID corrections (in degrees)
                        error_roll = self.pid_roll.compute(filtered_roll, elapsed)
                        error_pitch = self.pid_pitch.compute(filtered_pitch, elapsed)
                        
                        # Convert to foot position shifts (meters)
                        # Pitch error -> X shift, Roll error -> Y shift
                        self.active_foot_shift_x = (error_pitch * 
                            config.TILT_TO_FOOT_SHIFT_GAIN * config.PITCH_TO_X_SIGN)
                        self.active_foot_shift_y = (error_roll * 
                            config.TILT_TO_FOOT_SHIFT_GAIN * config.ROLL_TO_Y_SIGN)
                        
                        # Clamp to limits
                        self.active_foot_shift_x = np.clip(
                            self.active_foot_shift_x, 
                            -config.MAX_FOOT_SHIFT_X, 
                            config.MAX_FOOT_SHIFT_X)
                        self.active_foot_shift_y = np.clip(
                            self.active_foot_shift_y, 
                            -config.MAX_FOOT_SHIFT_Y, 
                            config.MAX_FOOT_SHIFT_Y)
                        
                        if config.DEBUG_PRINT:
                            print(f"[Balance] {current_name} roll={filtered_roll:.1f}° pitch={filtered_pitch:.1f}° "
                                  f"shift_x={self.active_foot_shift_x*1000:.1f}mm "
                                  f"shift_y={self.active_foot_shift_y*1000:.1f}mm")
                    elif current_name == 'return_to_stand':
                        # Reset corrections when returning to stand
                        self.active_foot_shift_x = 0.0
                        self.active_foot_shift_y = 0.0
                
                # Run movement
                complete = self.run_movement_step()
                
                if complete:
                    print("[Balancer] Movement sequence complete")
                    break
            
        except Exception as e:
            print(f"[Balancer] Error during bench test: {e}")
            import traceback
            traceback.print_exc()
            self.emergency_stop.trigger(str(e))
        
        finally:
            self.stop()
        
        return True
    
    def run_physical_balance(self):
        """
        Run the full physical balance control loop.
        
        Robot will attempt to balance on two diagonal legs with active IMU feedback.
        """
        print("[Balancer] Starting physical balance test...")
        print("[Balancer] WARNING: Robot will attempt to balance!")
        print("[Balancer] Keep hands ready to catch!")
        print("[Balancer] Press Ctrl+C to abort")
        
        time.sleep(3)  # Give time to read warning
        
        if not self.initialize_hardware():
            print("[Balancer] Hardware initialization failed, aborting")
            return False
        
        # Calibrate baseline
        self.calibrate_baseline()
        
        # Generate initial movements
        self.generate_movements()
        
        # Start sensor fusion
        self.sensor_fusion.start()
        
        # Run the control loop (skip safety state machine for now - direct execution)
        self.running = True
        self.last_loop_time = time.time()
        self.previous_time = time.time()
        start_time = time.time()
        
        print("[Balancer] Entering control loop with active balance...")
        
        # Track balance phase
        in_balance_phase = False
        
        try:
            while self.running:
                now = time.time()
                if now - self.last_loop_time < config.CONTROL_LOOP_DT:
                    continue
                self.last_loop_time = now
                
                # Check timeout
                if now - start_time > config.TIMEOUT_S:
                    print(f"[Balancer] Timeout after {config.TIMEOUT_S}s, returning to stand")
                    break
                
                # Read IMU
                roll, pitch = self.read_imu()
                
                # Filter
                filtered_roll, filtered_pitch = self.sensor_fusion.update(roll, pitch)
                
                # Check safety abort
                if abs(filtered_roll) > config.MAX_TILT_ABORT or abs(filtered_pitch) > config.MAX_TILT_ABORT:
                    print(f"[Balancer] Tilt abort: roll={filtered_roll:.1f}°, pitch={filtered_pitch:.1f}°")
                    self.emergency_stop.trigger("Excessive tilt")
                    break
                
                # Check current movement phase
                if self.movement_scheme is not None:
                    current_name = self.movement_scheme.movements_now.getMovementName()
                    
                    # Detect any lift phase (gradual steps or final lift)
                    is_lift_phase = ('gradual_lift' in current_name or 
                                     current_name == 'diagonal_lift')
                    
                    if is_lift_phase and not in_balance_phase:
                        in_balance_phase = True
                        print("[Balancer] Entering active balance phase (gradual lift)...")
                    
                    # Apply PID corrections during any lift phase
                    if in_balance_phase and is_lift_phase:
                        elapsed = now - self.previous_time
                        self.previous_time = now
                        
                        # Compute PID corrections (in degrees)
                        error_roll = self.pid_roll.compute(filtered_roll, elapsed)
                        error_pitch = self.pid_pitch.compute(filtered_pitch, elapsed)
                        
                        # Convert to foot position shifts (meters)
                        self.active_foot_shift_x = (error_pitch * 
                            config.TILT_TO_FOOT_SHIFT_GAIN * config.PITCH_TO_X_SIGN)
                        self.active_foot_shift_y = (error_roll * 
                            config.TILT_TO_FOOT_SHIFT_GAIN * config.ROLL_TO_Y_SIGN)
                        
                        # Clamp to limits
                        self.active_foot_shift_x = np.clip(
                            self.active_foot_shift_x, 
                            -config.MAX_FOOT_SHIFT_X, 
                            config.MAX_FOOT_SHIFT_X)
                        self.active_foot_shift_y = np.clip(
                            self.active_foot_shift_y, 
                            -config.MAX_FOOT_SHIFT_Y, 
                            config.MAX_FOOT_SHIFT_Y)
                        
                        if config.DEBUG_PRINT:
                            print(f"[Balance] {current_name} roll={filtered_roll:.1f}° pitch={filtered_pitch:.1f}° "
                                  f"shift_x={self.active_foot_shift_x*1000:.1f}mm "
                                  f"shift_y={self.active_foot_shift_y*1000:.1f}mm")
                    elif current_name == 'return_to_stand':
                        # Reset corrections when returning to stand
                        self.active_foot_shift_x = 0.0
                        self.active_foot_shift_y = 0.0
                
                # Check if movement complete
                lib_length = len(self.movement_lib)
                current_idx = self.movement_scheme.movement_now_number
                tick = self.movement_scheme.tick
                now_ticks = self.movement_scheme.now_ticks
                movement_complete = (current_idx >= lib_length - 1 and tick >= now_ticks)
                
                if movement_complete:
                    print("[Balancer] Balance sequence complete")
                    break
                
                # Run movement step
                self.run_movement_step()
            
        except Exception as e:
            print(f"[Balancer] Error during balance: {e}")
            import traceback
            traceback.print_exc()
            self.emergency_stop.trigger(str(e))
        
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the balancer and clean up."""
        self.running = False
        
        print("[Balancer] Stopping...")
        
        # Reset balance corrections
        self.active_foot_shift_x = 0.0
        self.active_foot_shift_y = 0.0
        
        # Stop sensor fusion
        self.sensor_fusion.stop()
        
        # ALWAYS return to safe stance if hardware available (even after emergency stop)
        if self.hardware is not None:
            try:
                print("[Balancer] Returning to default stance...")
                # Generate return movement
                self.movement_gen.clear_movements()
                self.movement_gen.return_to_stand(time_return=0.5)
                
                if self.movement_scheme is not None:
                    self.movement_lib = self.movement_gen.MovementLib
                    self.movement_scheme = self._MovementScheme(self.movement_lib)
                    
                    # Execute return
                    for _ in range(100):  # Max iterations
                        complete = self.run_movement_step()
                        if complete:
                            break
                        time.sleep(0.01)
                
            except Exception as e:
                print(f"[Balancer] Error during return: {e}")
                # Fallback: try to set default stance directly
                self._emergency_return_to_stand()
        
        print("[Balancer] Stopped")
    
    def _emergency_return_to_stand(self):
        """Emergency fallback to return legs to default position."""
        if self.hardware is None:
            return
        
        try:
            print("[Balancer] Emergency return to stand...")
            # Use default stance from config
            default_pos = config.DEFAULT_STANCE_MATRIX.copy()
            
            # Compute joint angles directly
            if hasattr(self, '_four_legs_inverse_kinematics') and hasattr(self, 'config'):
                joint_angles = self._four_legs_inverse_kinematics(default_pos, self.config)
                self.hardware.set_actuator_postions(joint_angles)
                print("[Balancer] Legs reset to default stance")
        except Exception as e:
            print(f"[Balancer] Emergency return failed: {e}")
    
    def run(self):
        """Run the balancer in the configured mode."""
        if self.mode == "bench":
            return self.run_bench_test()
        elif self.mode == "physical":
            return self.run_physical_balance()
        else:
            print(f"[Balancer] Unknown mode: {self.mode}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diagonal Balance Experiment for Mini Pupper 2"
    )
    parser.add_argument(
        "--mode", 
        choices=["bench", "physical"],
        default="bench",
        help="Operating mode (default: bench)"
    )
    parser.add_argument(
        "--pair",
        choices=["FR_BL", "FL_BR"],
        default=config.DIAGONAL_PAIR,
        help="Diagonal pair to lift (default: from config)"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=config.LIFT_HEIGHT,
        help="Lift height in meters (default: from config)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.TIMEOUT_S,
        help="Timeout in seconds (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Update config from args
    config.DIAGONAL_PAIR = args.pair
    config.LIFT_HEIGHT = args.height
    config.TIMEOUT_S = args.timeout
    
    print("=" * 60)
    print("Diagonal Balance Experiment for Mini Pupper 2")
    print("=" * 60)
    print(f"Mode:    {args.mode}")
    print(f"Pair:    {args.pair}")
    print(f"Height:  {args.height} m")
    print(f"Timeout: {args.timeout} s")
    print("=" * 60)
    
    # Create and run balancer
    balancer = DiagonalBalancer(mode=args.mode)
    success = balancer.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
