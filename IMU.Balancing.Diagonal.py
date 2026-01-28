#
# Copyright(C) 2025 MangDang (www.mangdang.net) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Description: Diagonal balance control demo based on IMU sensor for Mini Pupper 2.
#              The robot lifts two diagonal legs while balancing on the other two.
#
# Test method:
#   $python IMU.Balancing.Diagonal.py
#   

import numpy as np
import time
from src.IMU import IMU
from src.Controller import Controller
from src.State import State
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from MangDang.mini_pupper.display import Display
from src.MovementScheme import MovementScheme
from src.Command import Command
from src.MovementGroup import MovementGroups
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import math
import csv
from threading import Thread
import queue
import argparse

# Initialize movement groups
Move = MovementGroups()

def add_diagonal_balance_movement(desired_roll_angle, desired_pitch_angle, diagonal='FR_BL', lift_height=0.03):
    """Add a diagonal balance correction movement to the queue.
    
    Args:
        desired_roll_angle: Target roll angle in degrees
        desired_pitch_angle: Target pitch angle in degrees
        diagonal: Which diagonal to lift ('FR_BL' or 'FL_BR')
        lift_height: Height to lift diagonal legs (meters)
    """
    Move.balance_diagonal(desired_roll_angle, desired_pitch_angle, diagonal, lift_height, 0.015, 0)


class IIRLowPassFilter:
    """IIR Low Pass Filter implementation with configurable order."""
    
    def __init__(self, cutoff_freq, sample_rate, order):
        """
        Initialize IIR low pass filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sampling frequency in Hz
            order: Filter order (1 or 2)
        """
        self.cutoff = cutoff_freq
        self.fs = sample_rate
        self.order = order
        
        # Calculate filter coefficients
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        if order == 1:
            # First order Butterworth coefficients
            self.b = [normal_cutoff, normal_cutoff]
            self.a = [1, normal_cutoff - 1]
        else:
            # Second order Butterworth coefficients
            sqrt2 = np.sqrt(2)
            self.b = [normal_cutoff**2, 2*normal_cutoff**2, normal_cutoff**2]
            self.a = [1, 2*(normal_cutoff**2 - 1), 1 - sqrt2*normal_cutoff + normal_cutoff**2]
            
        self.x_hist = [0] * (order + 1)
        self.y_hist = [0] * (order + 1)
    
    def update(self, new_value):
        """
        Update the filter with a new measurement.
        
        Returns:
            The filtered value
        """
        # Shift history
        self.x_hist.pop()
        self.x_hist.insert(0, new_value)
        self.y_hist.pop()
        
        # Compute new output
        y = 0
        for i in range(len(self.b)):
            y += self.b[i] * self.x_hist[i]
        for i in range(1, len(self.a)):
            y -= self.a[i] * self.y_hist[i-1]
        
        y /= self.a[0]
        self.y_hist.insert(0, y)
        
        return y


class RealTimeEKF:
    """Thread-safe Extended Kalman Filter implementation for real-time sensor fusion."""
    
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        """
        Initialize the Real-Time EKF.
        
        Args:
            initial_state: Initial state vector [ax, ay, vx, vy]
            initial_covariance: Initial covariance matrix (4x4)
            process_noise: Process noise covariance matrix (4x4)
            measurement_noise: Measurement noise covariance matrix (2x2)
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.Q = process_noise
        self.R = measurement_noise
        self.last_time = time.time()
        
        # Thread-safe queue for incoming measurements
        self.measurement_queue = queue.Queue()
        
        # Flag for controlling the filter thread
        self.running = False
        self.filter_thread = None
    
    def predict(self, current_time=None):
        """Perform prediction step of the EKF."""
        if current_time is None:
            current_time = time.time()
            
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # State transition matrix
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Process noise matrix (scaled by dt)
        Q = self.Q * dt
        
        # Predict state and covariance
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
    
    def update(self, measurement):
        """Perform update step of the EKF."""
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        # Calculate Kalman gain
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        measurement_residual = measurement - H @ self.state
        self.state = self.state + K @ measurement_residual
        self.covariance = (np.eye(4) - K @ H) @ self.covariance
    
    def filter_loop(self):
        """Main filtering loop to run in a separate thread."""
        while self.running:
            try:
                # Get the latest measurement with timeout
                measurement = self.measurement_queue.get(timeout=0.1)
                self.filter_measurement(measurement)
            except queue.Empty:
                continue
                
    def filter_measurement(self, measurement):
        """Process a single measurement through the EKF."""
        current_time = time.time()
        self.predict(current_time)
        self.update(measurement)
        return self.state[:2]  # Return filtered ax, ay
        
    def start(self):
        """Start the real-time filtering thread."""
        if not self.running:
            self.running = True
            self.filter_thread = Thread(target=self.filter_loop)
            self.filter_thread.start()
            
    def stop(self):
        """Stop the filtering thread."""
        self.running = False
        if self.filter_thread is not None:
            self.filter_thread.join()
            
    def add_measurement(self, ax, ay):
        """Add a new measurement to the queue (thread-safe)."""
        self.measurement_queue.put(np.array([ax, ay]))
        
    def get_filtered_output(self):
        """Get the current filtered state (thread-safe)."""
        return self.state[:2].copy()  # Return a copy to avoid thread issues


class PIDController:
    """Standard PID controller implementation."""
    
    def __init__(self, Kp, Ki, Kd, setpoint):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            setpoint: Target value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, process_variable, dt):
        """
        Compute PID output.
        
        Args:
            process_variable: Current measured value
            dt: Time step since last computation
            
        Returns:
            PID control output
        """
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error
        
        return output


def angle_converter_roll(ax, ay, az):
    """Convert accelerometer readings to roll angle in degrees."""
    return math.degrees(math.atan(ax / math.sqrt(ay * ay + az * az)))


def angle_converter_pitch(ax, ay, az):
    """Convert accelerometer readings to pitch angle in degrees."""
    return math.degrees(math.atan(ay / math.sqrt(ax * ax + az * az)))


def main(use_imu=False, diagonal='FR_BL', lift_height=0.03, log_data=False):
    """Main control loop for diagonal balance on the quadruped robot.
    
    Args:
        use_imu: Boolean flag to enable/disable external IMU usage
        diagonal: Which diagonal to lift ('FR_BL' or 'FL_BR')
        lift_height: Height to lift the diagonal legs (meters)
        log_data: Boolean flag to enable/disable CSV logging
    """
    print(f"Starting Diagonal Balance Demo")
    print(f"  Diagonal to lift: {diagonal}")
    print(f"  Lift height: {lift_height}m")
    print(f"  Press Ctrl+C to exit")
    print("-" * 40)
    
    # Create config
    config = Configuration()
    hardware_interface = HardwareInterface()
    disp = Display()
    disp.show_ip()

    # Create imu handle
    if use_imu:
        imu = IMU(port="/dev/ttyACM0")
        imu.flush_buffer()
    esp32 = ESP32Interface()

    # Create controller and user input handles
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()

    # Create movement group scheme instance with initial diagonal balance pose
    Move.balance_diagonal(0, 0, diagonal, lift_height, 0.5, 0.3)  # Initial pose with transition time
    MovementLib = Move.MovementLib
    movementCtl = MovementScheme(MovementLib)
    lib_length = len(MovementLib)
    
    last_loop = time.time()

    command = Command()
    command.pseudo_dance_event = True

    # Measure original roll and pitch angle (reference for balance)
    original = esp32.imu_get_data()
    original_roll = angle_converter_roll(original['ax'], original['ay'], original['az'])
    original_pitch = angle_converter_pitch(original['ax'], original['ay'], original['az'])
    
    print(f"Reference angles - Roll: {original_roll:.2f}°, Pitch: {original_pitch:.2f}°")

    # Create PID controllers
    # Tuned for diagonal balance - may need adjustment
    kp = 1.0   # Slightly higher P gain for faster response with fewer legs
    ki = 0.02
    kd = 0.02
    pid_roll = PIDController(kp, ki, kd, original_roll)
    pid_pitch = PIDController(kp, ki, kd, original_pitch)

    # Low-pass filter initializer
    sample_rate = 1 / 0.005  # Hz (200 Hz)
    cutoff_freq = 6   # Hz
    ax_filtered_lowpass = IIRLowPassFilter(cutoff_freq, sample_rate, order=1)
    ay_filtered_lowpass = IIRLowPassFilter(cutoff_freq, sample_rate, order=1)

    # Kalman filter initializer
    initial_state = np.array([0, 0, 0, 0])  # [ax, ay, vx, vy]
    initial_covariance = np.eye(4) * 0.1
    process_noise = np.eye(4) * 0.01
    measurement_noise = np.eye(2) * 0.1

    ekf = RealTimeEKF(initial_state, initial_covariance, process_noise, measurement_noise)
    ekf.start()  # Start the filtering thread

    # CSV logging setup
    log_file = None
    csv_writer = None
    if log_data:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/diagonal_balance_log_{timestamp}.csv"
        log_file = open(log_filename, 'w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['time', 'raw_roll', 'raw_pitch', 'filtered_roll', 'filtered_pitch', 
                            'error_roll', 'error_pitch', 'diagonal'])
        print(f"Logging to: {log_filename}")

    # Other Variable Initializer
    previous_time = time.time()
    start_time = time.time()
    error_roll = 0
    error_pitch = 0 

    try:
        # Main control loop
        while True:
            now = time.time()
            if now - last_loop < 0.005:  # 200Hz control loop
                continue
            last_loop = time.time()

            # Read imu data
            quat_orientation = (
                imu.read_orientation() if use_imu else np.array([1, 0, 0, 0])
            )
            state.quat_orientation = quat_orientation
            
            # IMU data processing
            imu_data = esp32.imu_get_data()
            roll_angle = angle_converter_roll(imu_data['ax'], imu_data['ay'], imu_data['az'])
            pitch_angle = angle_converter_pitch(imu_data['ax'], imu_data['ay'], imu_data['az'])
            
            # Sensor fusion pipeline
            ekf.add_measurement(ax_filtered_lowpass.update(roll_angle), 
                               ay_filtered_lowpass.update(pitch_angle))
            filtered = ekf.get_filtered_output()
            filtered_roll_angle = filtered[0]
            filtered_pitch_angle = filtered[1]
            
            # Balance correction logic
            # Threshold for triggering balance correction
            if (abs(filtered_roll_angle - original_roll) > 0.5 or 
                abs(filtered_pitch_angle - original_pitch) > 0.5):
                
                if movementCtl.movement_now_number >= lib_length - 1 and movementCtl.tick >= movementCtl.now_ticks:
                    elapsed = time.time() - previous_time
                    if elapsed > 0:
                        # Calculate driven angles by PID controller
                        error_roll = pid_roll.compute(filtered_roll_angle, elapsed)
                        error_pitch = pid_pitch.compute(-filtered_pitch_angle, elapsed)
                        previous_time = time.time()
                        
                        # Create the diagonal balance movement
                        add_diagonal_balance_movement(error_roll, error_pitch, diagonal, lift_height)
                        
                        # Log data
                        if csv_writer:
                            csv_writer.writerow([
                                time.time() - start_time,
                                roll_angle, pitch_angle,
                                filtered_roll_angle, filtered_pitch_angle,
                                error_roll, error_pitch,
                                diagonal
                            ])
            
            # Movement control
            movementCtl.runMovementScheme()
            command.legslocation = movementCtl.getMovemenLegsLocation()
            command.horizontal_velocity = movementCtl.getMovemenSpeed()
            command.roll = movementCtl.attitude_now[0]
            command.pitch = movementCtl.attitude_now[1]
            command.yaw = movementCtl.attitude_now[2]
            command.yaw_rate = movementCtl.getMovemenTurn()
            
            # Get the static legs mask from the current movement
            # This tells the Controller which legs should NOT rotate with body attitude
            command.static_legs_mask = movementCtl.getStaticLegsMask()
            
            # Run controller and update hardware
            controller.run(state, command, disp)
            hardware_interface.set_actuator_postions(state.joint_angles)
            command = Command()
            
    except KeyboardInterrupt:
        print("\nStopping diagonal balance demo...")
    finally:
        ekf.stop()
        if log_file:
            log_file.close()
            print(f"Log saved.")
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagonal Balance Demo for Mini Pupper 2')
    parser.add_argument('--diagonal', type=str, default='FR_BL', choices=['FR_BL', 'FL_BR'],
                       help='Which diagonal to lift: FR_BL (Front-Right + Back-Left) or FL_BR (Front-Left + Back-Right)')
    parser.add_argument('--lift-height', type=float, default=0.03,
                       help='Height to lift diagonal legs in meters (default: 0.03)')
    parser.add_argument('--use-imu', action='store_true',
                       help='Enable external IMU')
    parser.add_argument('--log', action='store_true',
                       help='Enable CSV data logging')
    
    args = parser.parse_args()
    
    main(use_imu=args.use_imu, diagonal=args.diagonal, lift_height=args.lift_height, log_data=args.log)
