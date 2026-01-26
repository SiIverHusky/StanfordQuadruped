"""
Sensor Filters for Diagonal Balance Experiment.

Contains:
- IIR Low-Pass Filter
- Extended Kalman Filter (EKF)
- PID Controller

These are standalone implementations that do not modify the original
IMU.Balancing.MP2.py code.
"""

import numpy as np
import time
import queue
from threading import Thread
from experiments.diagonal_balance import config


class IIRLowPassFilter:
    """
    IIR Low Pass Filter implementation with configurable order.
    
    Copied from IMU.Balancing.MP2.py and isolated for the experiment.
    """
    
    def __init__(self, cutoff_freq=None, sample_rate=None, order=None):
        """
        Initialize IIR low pass filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz (default from config)
            sample_rate: Sampling frequency in Hz (default from config)
            order: Filter order, 1 or 2 (default from config)
        """
        cutoff_freq = cutoff_freq if cutoff_freq is not None else config.FILTER_CUTOFF_FREQ
        sample_rate = sample_rate if sample_rate is not None else config.FILTER_SAMPLE_RATE
        order = order if order is not None else config.FILTER_ORDER
        
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
        
        Args:
            new_value: New input value
        
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
    
    def reset(self):
        """Reset filter history."""
        self.x_hist = [0] * (self.order + 1)
        self.y_hist = [0] * (self.order + 1)


class RealTimeEKF:
    """
    Thread-safe Extended Kalman Filter implementation for real-time sensor fusion.
    
    Copied from IMU.Balancing.MP2.py and isolated for the experiment.
    """
    
    def __init__(self, initial_state=None, initial_covariance=None, 
                 process_noise=None, measurement_noise=None):
        """
        Initialize the Real-Time EKF.
        
        Args:
            initial_state: Initial state vector [ax, ay, vx, vy] (default from config)
            initial_covariance: Initial covariance matrix 4x4 (default from config)
            process_noise: Process noise covariance matrix 4x4 (default from config)
            measurement_noise: Measurement noise covariance matrix 2x2 (default from config)
        """
        self.state = initial_state if initial_state is not None else config.EKF_INITIAL_STATE.copy()
        self.covariance = initial_covariance if initial_covariance is not None else config.EKF_INITIAL_COVARIANCE.copy()
        self.Q = process_noise if process_noise is not None else config.EKF_PROCESS_NOISE.copy()
        self.R = measurement_noise if measurement_noise is not None else config.EKF_MEASUREMENT_NOISE.copy()
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
            self.filter_thread = Thread(target=self.filter_loop, daemon=True)
            self.filter_thread.start()
            
    def stop(self):
        """Stop the filtering thread."""
        self.running = False
        if self.filter_thread is not None:
            self.filter_thread.join(timeout=1.0)
            self.filter_thread = None
            
    def add_measurement(self, ax, ay):
        """Add a new measurement to the queue (thread-safe)."""
        self.measurement_queue.put(np.array([ax, ay]))
        
    def get_filtered_output(self):
        """Get the current filtered state (thread-safe)."""
        return self.state[:2].copy()  # Return a copy to avoid thread issues
    
    def reset(self):
        """Reset the EKF state."""
        self.state = config.EKF_INITIAL_STATE.copy()
        self.covariance = config.EKF_INITIAL_COVARIANCE.copy()
        self.last_time = time.time()
        
        # Clear the measurement queue
        while not self.measurement_queue.empty():
            try:
                self.measurement_queue.get_nowait()
            except queue.Empty:
                break


class PIDController:
    """
    Standard PID controller implementation.
    
    Copied from IMU.Balancing.MP2.py and isolated for the experiment.
    """
    
    def __init__(self, Kp=None, Ki=None, Kd=None, setpoint=0.0):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain (default from config)
            Ki: Integral gain (default from config)
            Kd: Derivative gain (default from config)
            setpoint: Target value
        """
        self.Kp = Kp if Kp is not None else config.PID_KP
        self.Ki = Ki if Ki is not None else config.PID_KI
        self.Kd = Kd if Kd is not None else config.PID_KD
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        
        # Output limits
        self.output_min = config.PID_OUTPUT_MIN
        self.output_max = config.PID_OUTPUT_MAX
        
        # Integral limits (anti-windup)
        self.integral_min = config.PID_INTEGRAL_MIN
        self.integral_max = config.PID_INTEGRAL_MAX

    def compute(self, process_variable, dt):
        """
        Compute PID output.
        
        Args:
            process_variable: Current measured value
            dt: Time step since last computation
            
        Returns:
            PID control output (clamped to limits)
        """
        if dt <= 0:
            dt = 0.001  # Prevent division by zero
        
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(self.integral_min, min(self.integral_max, self.integral))
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Clamp output
        output = max(self.output_min, min(self.output_max, output))
        
        # Update previous error
        self.previous_error = error
        
        return output
    
    def reset(self):
        """Reset the PID controller state."""
        self.previous_error = 0
        self.integral = 0
    
    def set_setpoint(self, setpoint):
        """Update the setpoint."""
        self.setpoint = setpoint
    
    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Update PID gains."""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd


class SensorFusion:
    """
    Combined sensor fusion pipeline.
    
    Integrates IIR filters and EKF for roll/pitch angle processing.
    """
    
    def __init__(self):
        """Initialize the sensor fusion pipeline."""
        # IIR filters for roll and pitch
        self.roll_filter = IIRLowPassFilter()
        self.pitch_filter = IIRLowPassFilter()
        
        # EKF for combined filtering
        self.ekf = RealTimeEKF()
        
        # Store last filtered values
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
    
    def start(self):
        """Start the EKF background thread."""
        self.ekf.start()
    
    def stop(self):
        """Stop the EKF background thread."""
        self.ekf.stop()
    
    def update(self, roll_deg, pitch_deg):
        """
        Update the sensor fusion with new angle measurements.
        
        Args:
            roll_deg: Raw roll angle in degrees
            pitch_deg: Raw pitch angle in degrees
            
        Returns:
            Tuple of (filtered_roll, filtered_pitch) in degrees
        """
        # First stage: IIR low-pass filter
        iir_roll = self.roll_filter.update(roll_deg)
        iir_pitch = self.pitch_filter.update(pitch_deg)
        
        # Second stage: EKF
        self.ekf.add_measurement(iir_roll, iir_pitch)
        filtered = self.ekf.get_filtered_output()
        
        self.filtered_roll = filtered[0]
        self.filtered_pitch = filtered[1]
        
        return self.filtered_roll, self.filtered_pitch
    
    def get_filtered_angles(self):
        """Get the last filtered angles."""
        return self.filtered_roll, self.filtered_pitch
    
    def reset(self):
        """Reset all filters."""
        self.roll_filter.reset()
        self.pitch_filter.reset()
        self.ekf.reset()
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0


def angle_from_accel_roll(ax, ay, az):
    """
    Convert accelerometer readings to roll angle in degrees.
    
    Args:
        ax, ay, az: Accelerometer readings
        
    Returns:
        Roll angle in degrees
    """
    import math
    denom = math.sqrt(ay * ay + az * az)
    if denom < 1e-6:
        return 0.0
    return math.degrees(math.atan(ax / denom))


def angle_from_accel_pitch(ax, ay, az):
    """
    Convert accelerometer readings to pitch angle in degrees.
    
    Args:
        ax, ay, az: Accelerometer readings
        
    Returns:
        Pitch angle in degrees
    """
    import math
    denom = math.sqrt(ax * ax + az * az)
    if denom < 1e-6:
        return 0.0
    return math.degrees(math.atan(ay / denom))


if __name__ == "__main__":
    # Test filters
    print("Testing Filters...")
    
    # Test IIR filter
    print("\n1. IIR Low-Pass Filter:")
    iir = IIRLowPassFilter()
    noisy_signal = [1.0 + 0.5 * np.sin(i * 0.5) + 0.2 * np.random.randn() for i in range(20)]
    filtered_signal = [iir.update(x) for x in noisy_signal]
    print(f"   Input:  {[f'{x:.2f}' for x in noisy_signal[:5]]}...")
    print(f"   Output: {[f'{x:.2f}' for x in filtered_signal[:5]]}...")
    
    # Test PID
    print("\n2. PID Controller:")
    pid = PIDController(setpoint=0.0)
    errors = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2]
    outputs = [pid.compute(e, 0.01) for e in errors]
    print(f"   Errors:  {errors}")
    print(f"   Outputs: {[f'{o:.2f}' for o in outputs]}")
    
    # Test EKF
    print("\n3. Extended Kalman Filter:")
    ekf = RealTimeEKF()
    ekf.start()
    for i in range(10):
        ekf.add_measurement(1.0 + 0.1 * np.random.randn(), 0.5 + 0.1 * np.random.randn())
        time.sleep(0.01)
    output = ekf.get_filtered_output()
    print(f"   Filtered output: roll={output[0]:.3f}, pitch={output[1]:.3f}")
    ekf.stop()
    
    # Test sensor fusion
    print("\n4. Sensor Fusion Pipeline:")
    fusion = SensorFusion()
    fusion.start()
    for i in range(10):
        roll = 2.0 + 0.5 * np.random.randn()
        pitch = 1.0 + 0.3 * np.random.randn()
        filtered_roll, filtered_pitch = fusion.update(roll, pitch)
        time.sleep(0.01)
    print(f"   Final filtered: roll={filtered_roll:.3f}, pitch={filtered_pitch:.3f}")
    fusion.stop()
    
    print("\nAll filter tests passed!")
