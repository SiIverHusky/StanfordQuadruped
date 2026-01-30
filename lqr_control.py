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
# AUTHORS OR COPYRIGHT HOLDERSERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Description: This is a PID control demo based on IMU sensor ONLY for Mini Pupper 2.
#
# Test method:
#   $python IMU.Balancing.MP2.py
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

from scipy.linalg import solve_discrete_are
import argparse

# Initialize movement groups
Move = MovementGroups()

# Global diagonal configuration (set from command line args)
DIAGONAL_PAIR = 'FR_BL'
LIFT_HEIGHT = 0.03

def add_movement(desired_roll_angle, desired_pitch_angle):
    """Add a diagonal balance correction movement to the queue.
    
    Args:
        desired_roll_angle: Target roll angle in degrees
        desired_pitch_angle: Target pitch angle in degrees
    """
    Move.balance_diagonal(desired_roll_angle, desired_pitch_angle, DIAGONAL_PAIR, LIFT_HEIGHT, 0.015, 0)

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


class LQRController:
    """LQR controller for balancing roll/pitch angles."""

    def __init__(self, A, B, Q, R):
        """
        Initialize LQR controller.
        
        Args:
            A: State transition matrix
            B: Control input matrix
            Q: State cost matrix (Weight for state)
            R: Control cost matrix (Weight for control effort)
        """
        # Solve the discrete-time algebraic Riccati equation
        P = solve_discrete_are(A, B, Q, R)
        # Compute optimal feedback gain
        self.K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)

    def compute(self, state):
        """
        Compute LQR control input.
        
        Args:
            state: Current state vector [angle, angular_velocity]
        
        Returns:
            Control output (desired correction)
        """
        u = -self.K @ state
        return u[0], u[1]


class StanceFootShifter:
    """
    Computes foot position shifts for stance legs to counteract tipping.
    
    Tipping correction strategy (world frame reference):
    - Forward tipping (positive pitch): Move front foot back, rear foot forward
    - Backward tipping (negative pitch): Move front foot forward, rear foot back
    - Left roll (positive roll): Shift both feet right
    - Right roll (negative roll): Shift both feet left
    
    Leg indices:
        Leg 0: Front-Right [0.06, -0.05, -0.07]
        Leg 1: Front-Left  [0.06,  0.05, -0.07]
        Leg 2: Back-Right  [-0.06, -0.05, -0.07]
        Leg 3: Back-Left   [-0.06,  0.05, -0.07]
    """
    
    # Default stance positions for each leg [x, y, z]
    DEFAULT_STANCE = {
        0: [0.06, -0.05, -0.07],   # Front-Right
        1: [0.06, 0.05, -0.07],    # Front-Left
        2: [-0.06, -0.05, -0.07],  # Back-Right
        3: [-0.06, 0.05, -0.07],   # Back-Left
    }
    
    # Diagonal pair configurations
    DIAGONAL_CONFIG = {
        'FR_BL': {
            'lifted': [0, 3],    # Front-Right, Back-Left
            'stance': [1, 2],    # Front-Left, Back-Right
            'front_stance': 1,   # Front-Left is the front stance leg
            'rear_stance': 2,    # Back-Right is the rear stance leg
        },
        'FL_BR': {
            'lifted': [1, 2],    # Front-Left, Back-Right
            'stance': [0, 3],    # Front-Right, Back-Left
            'front_stance': 0,   # Front-Right is the front stance leg
            'rear_stance': 3,    # Back-Left is the rear stance leg
        }
    }
    
    def __init__(self, diagonal='FR_BL', 
                 pitch_to_x_gain=0.002,   # meters per degree of pitch error
                 roll_to_y_gain=0.002,    # meters per degree of roll error
                 max_x_shift=0.03,        # max X shift in meters
                 max_y_shift=0.02,        # max Y shift in meters
                 invert_x=False,          # flip X shift direction
                 invert_y=False):         # flip Y shift direction
        """
        Initialize the stance foot shifter.
        
        Args:
            diagonal: Which diagonal is lifted ('FR_BL' or 'FL_BR')
            pitch_to_x_gain: How much to shift X per degree of pitch error
            roll_to_y_gain: How much to shift Y per degree of roll error
            max_x_shift: Maximum allowed X shift (meters)
            max_y_shift: Maximum allowed Y shift (meters)
            invert_x: If True, flip the X shift direction (front/back)
            invert_y: If True, flip the Y shift direction (left/right)
        """
        self.diagonal = diagonal
        self.config = self.DIAGONAL_CONFIG[diagonal]
        self.pitch_to_x_gain = pitch_to_x_gain
        self.roll_to_y_gain = roll_to_y_gain
        self.max_x_shift = max_x_shift
        self.max_y_shift = max_y_shift
        self.invert_x = invert_x
        self.invert_y = invert_y
        
        # Current shift values
        self.x_shift_front = 0.0
        self.x_shift_rear = 0.0
        self.y_shift = 0.0
    
    def compute_shifts(self, pitch_error_deg, roll_error_deg):
        """
        Compute foot position shifts based on tipping errors.
        
        Args:
            pitch_error_deg: Pitch error in degrees (positive = forward tipping)
            roll_error_deg: Roll error in degrees (positive = left tipping)
            
        Returns:
            Tuple of (x_shift_front, x_shift_rear, y_shift) in meters
        """
        # Direction multipliers based on invert flags
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1
        
        # Pitch correction: forward tipping -> front foot forward, rear foot back
        # pitch_error > 0 means tipping forward
        # Front foot: move forward (positive X) to counteract forward tip
        # Rear foot: move back (negative X) to counteract forward tip
        self.x_shift_front = x_dir * pitch_error_deg * self.pitch_to_x_gain
        self.x_shift_rear = x_dir * -pitch_error_deg * self.pitch_to_x_gain
        
        # Roll correction: left tipping -> shift feet left (positive Y)
        # roll_error > 0 means tipping left
        # Both feet: shift left (positive Y) to counteract left tip
        self.y_shift = y_dir * roll_error_deg * self.roll_to_y_gain
        
        # Clamp to limits
        self.x_shift_front = np.clip(self.x_shift_front, -self.max_x_shift, self.max_x_shift)
        self.x_shift_rear = np.clip(self.x_shift_rear, -self.max_x_shift, self.max_x_shift)
        self.y_shift = np.clip(self.y_shift, -self.max_y_shift, self.max_y_shift)
        
        return self.x_shift_front, self.x_shift_rear, self.y_shift
    
    def apply_shifts_to_legslocation(self, legslocation):
        """
        Apply the computed shifts to legslocation.
        
        Args:
            legslocation: Current leg positions [[x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]]
            
        Returns:
            Modified legslocation with shifts applied to stance legs
        """
        front_leg = self.config['front_stance']
        rear_leg = self.config['rear_stance']
        
        # Apply X shifts (different for front and rear)
        legslocation[0][front_leg] = self.DEFAULT_STANCE[front_leg][0] + self.x_shift_front
        legslocation[0][rear_leg] = self.DEFAULT_STANCE[rear_leg][0] + self.x_shift_rear
        
        # Apply Y shifts (same for both stance legs)
        legslocation[1][front_leg] = self.DEFAULT_STANCE[front_leg][1] + self.y_shift
        legslocation[1][rear_leg] = self.DEFAULT_STANCE[rear_leg][1] + self.y_shift
        
        return legslocation
    
    def get_shift_info(self):
        """Return current shift values for debugging."""
        return {
            'x_shift_front': self.x_shift_front,
            'x_shift_rear': self.x_shift_rear,
            'y_shift': self.y_shift,
            'front_leg': self.config['front_stance'],
            'rear_leg': self.config['rear_stance'],
        }


class TorqueBasedBalanceController:
    """
    Computes required foot position adjustments based on torque balance analysis.
    
    This controller calculates the ground reaction forces needed at each stance foot
    to maintain balance, then determines foot position shifts to achieve those forces.
    
    The key insight is that shifting a foot position changes the moment arm,
    which changes the torque that foot can apply to the body.
    
    Coordinate system (body frame):
        X: Forward (+) / Backward (-)
        Y: Left (+) / Right (-)
        Z: Up (+) / Down (-)
    
    Leg indices:
        0: Front-Right (FR)
        1: Front-Left (FL)
        2: Back-Right (BR)
        3: Back-Left (BL)
    """
    
    # Mini Pupper 2 physical parameters
    ROBOT_MASS = 0.4  # kg (approximate)
    GRAVITY = 9.81    # m/s^2
    
    # Link lengths for Mini Pupper 2 (meters)
    L1 = 0.0125   # Hip offset (lateral)
    L2 = 0.04     # Upper leg length
    L3 = 0.055    # Lower leg length
    
    # Default stance positions [x, y, z] in body frame (meters)
    DEFAULT_STANCE = {
        0: np.array([0.06, -0.05, -0.07]),   # Front-Right
        1: np.array([0.06, 0.05, -0.07]),    # Front-Left
        2: np.array([-0.06, -0.05, -0.07]),  # Back-Right
        3: np.array([-0.06, 0.05, -0.07]),   # Back-Left
    }
    
    # Center of mass offset from body center [x, y, z] (meters)
    # Adjust based on actual robot - may be slightly forward due to head/electronics
    COM_OFFSET = np.array([0.0, 0.0, 0.0])
    
    # Approximate moments of inertia (kg*m^2)
    I_ROLL = 0.001   # About X axis
    I_PITCH = 0.002  # About Y axis
    
    # Diagonal configurations
    DIAGONAL_CONFIG = {
        'FR_BL': {
            'stance': [1, 2],    # Front-Left, Back-Right
            'front_stance': 1,
            'rear_stance': 2,
        },
        'FL_BR': {
            'stance': [0, 3],    # Front-Right, Back-Left
            'front_stance': 0,
            'rear_stance': 3,
        }
    }
    
    def __init__(self, diagonal='FR_BL', 
                 max_x_shift=0.025,
                 max_y_shift=0.02,
                 force_to_position_gain=0.0005,
                 kp_roll=50.0,
                 kp_pitch=50.0,
                 kd_roll=5.0,
                 kd_pitch=5.0,
                 invert_x=False,
                 invert_y=False):
        """
        Initialize the torque-based balance controller.
        
        Args:
            diagonal: Which diagonal is lifted ('FR_BL' or 'FL_BR')
            max_x_shift: Maximum X position shift (meters)
            max_y_shift: Maximum Y position shift (meters)
            force_to_position_gain: Gain for converting force error to position shift
            kp_roll: Proportional gain for roll correction
            kp_pitch: Proportional gain for pitch correction
            kd_roll: Derivative gain for roll damping
            kd_pitch: Derivative gain for pitch damping
            invert_x: Flip X shift direction
            invert_y: Flip Y shift direction
        """
        self.diagonal = diagonal
        self.config = self.DIAGONAL_CONFIG[diagonal]
        self.max_x_shift = max_x_shift
        self.max_y_shift = max_y_shift
        self.force_to_position_gain = force_to_position_gain
        self.kp_roll = kp_roll
        self.kp_pitch = kp_pitch
        self.kd_roll = kd_roll
        self.kd_pitch = kd_pitch
        self.invert_x = invert_x
        self.invert_y = invert_y
        
        # Get stance leg indices
        self.front_leg = self.config['front_stance']
        self.rear_leg = self.config['rear_stance']
        
        # Current foot positions (initialize to default)
        self.foot_positions = {
            self.front_leg: self.DEFAULT_STANCE[self.front_leg].copy(),
            self.rear_leg: self.DEFAULT_STANCE[self.rear_leg].copy(),
        }
        
        # Computed values for debugging
        self.desired_forces = {}
        self.actual_torques = np.zeros(3)
        self.required_torques = np.zeros(3)
        self.position_shifts = {'front': np.zeros(2), 'rear': np.zeros(2)}
        
    def compute_body_rotation_matrix(self, roll_rad, pitch_rad, yaw_rad=0):
        """
        Compute rotation matrix from body frame to world frame.
        
        Uses ZYX Euler angle convention (yaw-pitch-roll).
        
        Args:
            roll_rad: Roll angle in radians
            pitch_rad: Pitch angle in radians
            yaw_rad: Yaw angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R
    
    def compute_foot_positions_world(self, roll_rad, pitch_rad, foot_positions_body):
        """
        Transform foot positions from body frame to world frame.
        
        Args:
            roll_rad: Current roll angle
            pitch_rad: Current pitch angle
            foot_positions_body: Dict of leg_idx -> [x, y, z] in body frame
            
        Returns:
            Dict of leg_idx -> [x, y, z] in world frame
        """
        R = self.compute_body_rotation_matrix(roll_rad, pitch_rad)
        
        foot_positions_world = {}
        for leg_idx, pos_body in foot_positions_body.items():
            foot_positions_world[leg_idx] = R @ pos_body
            
        return foot_positions_world
    
    def compute_com_position(self, roll_rad, pitch_rad):
        """
        Compute Center of Mass position in world frame.
        
        Args:
            roll_rad: Current roll angle
            pitch_rad: Current pitch angle
            
        Returns:
            CoM position [x, y, z] in world frame
        """
        R = self.compute_body_rotation_matrix(roll_rad, pitch_rad)
        com_world = R @ self.COM_OFFSET
        return com_world
    
    def compute_required_ground_reaction_forces(self, roll_rad, pitch_rad, 
                                                 roll_rate=0, pitch_rate=0,
                                                 desired_roll_accel=0, desired_pitch_accel=0):
        """
        Compute the ground reaction forces needed at each stance foot for balance.
        
        For static balance with 2 feet, we need:
        1. Sum of vertical forces = mg (support weight)
        2. Sum of torques about CoM = desired angular acceleration * I
        
        For dynamic balance, torques should produce desired angular accelerations
        to correct the tipping.
        
        Args:
            roll_rad: Current roll angle
            pitch_rad: Current pitch angle
            roll_rate: Current roll angular velocity (rad/s)
            pitch_rate: Current pitch angular velocity (rad/s)
            desired_roll_accel: Desired roll angular acceleration (rad/s^2)
            desired_pitch_accel: Desired pitch angular acceleration (rad/s^2)
            
        Returns:
            Dict of leg_idx -> [Fx, Fy, Fz] ground reaction forces
        """
        mg = self.ROBOT_MASS * self.GRAVITY
        
        # Get foot positions in world frame
        foot_pos_world = self.compute_foot_positions_world(
            roll_rad, pitch_rad, self.foot_positions
        )
        
        # CoM position
        com_pos = self.compute_com_position(roll_rad, pitch_rad)
        
        # Position vectors from CoM to each foot
        r_front = foot_pos_world[self.front_leg] - com_pos
        r_rear = foot_pos_world[self.rear_leg] - com_pos
        
        # For quasi-static balance with 2 legs:
        # Vertical force distribution based on position along support line
        support_vector = r_rear - r_front
        support_length = np.linalg.norm(support_vector[:2])
        
        if support_length < 0.001:
            # Feet too close, equal distribution
            F_front_z = mg / 2
            F_rear_z = mg / 2
        else:
            # Project CoM onto support line to find force distribution
            t = -np.dot(r_front[:2], support_vector[:2]) / (support_length ** 2)
            t = np.clip(t, 0.1, 0.9)  # Keep some load on both feet
            
            F_rear_z = t * mg
            F_front_z = (1 - t) * mg
        
        # Required torques for desired angular accelerations
        tau_roll_required = self.I_ROLL * desired_roll_accel
        tau_pitch_required = self.I_PITCH * desired_pitch_accel
        
        # Current torques from vertical forces
        # τ_roll (about X) = y * Fz
        # τ_pitch (about Y) = -x * Fz
        tau_roll_current = r_front[1] * F_front_z + r_rear[1] * F_rear_z
        tau_pitch_current = -r_front[0] * F_front_z - r_rear[0] * F_rear_z
        
        # Torque error
        tau_roll_error = tau_roll_required - tau_roll_current
        tau_pitch_error = tau_pitch_required - tau_pitch_current
        
        # Compute horizontal forces needed to generate corrective torques
        # Using height of CoM above ground for lever arm
        h = abs(self.DEFAULT_STANCE[self.front_leg][2])
        
        if h > 0.01:
            # Horizontal forces for torque correction
            # Roll torque from Fy: τ_roll ≈ h * Fy
            # Pitch torque from Fx: τ_pitch ≈ h * Fx
            F_front_y = tau_roll_error / (2 * h)
            F_rear_y = tau_roll_error / (2 * h)
            
            F_front_x = -tau_pitch_error / (2 * h)
            F_rear_x = tau_pitch_error / (2 * h)
        else:
            F_front_y = F_rear_y = 0
            F_front_x = F_rear_x = 0
        
        self.desired_forces = {
            self.front_leg: np.array([F_front_x, F_front_y, F_front_z]),
            self.rear_leg: np.array([F_rear_x, F_rear_y, F_rear_z]),
        }
        
        self.actual_torques = np.array([tau_roll_current, tau_pitch_current, 0])
        self.required_torques = np.array([tau_roll_required, tau_pitch_required, 0])
        
        return self.desired_forces
    
    def compute_leg_jacobian(self, joint_angles, leg_idx):
        """
        Compute the Jacobian for a single leg.
        
        The Jacobian J relates joint velocities to foot velocity:
            v_foot = J @ θ̇
        
        And its transpose relates foot forces to joint torques:
            τ_joints = J^T @ F_foot
        
        Args:
            joint_angles: [θ1, θ2, θ3] for hip, thigh, shank
            leg_idx: Leg index (0-3)
            
        Returns:
            3x3 Jacobian matrix
        """
        θ1, θ2, θ3 = joint_angles
        
        # Leg side factor (right legs have negative Y offset)
        side = -1 if leg_idx in [0, 2] else 1  # FR, BR are right side
        
        # Trigonometric values
        s1, c1 = np.sin(θ1), np.cos(θ1)
        s2, c2 = np.sin(θ2), np.cos(θ2)
        s23, c23 = np.sin(θ2 + θ3), np.cos(θ2 + θ3)
        
        L1, L2, L3 = self.L1, self.L2, self.L3
        
        # Jacobian matrix (simplified 3-DOF leg)
        J = np.array([
            [0, -L2*c2 - L3*c23, -L3*c23],
            [side * (L1*c1 + (L2*s2 + L3*s23)*s1), 
             side * (L2*c2 + L3*c23)*c1, 
             side * L3*c23*c1],
            [side * (L1*s1 - (L2*s2 + L3*s23)*c1),
             (L2*c2 + L3*c23)*s1,
             L3*c23*s1]
        ])
        
        return J
    
    def compute_joint_torques_from_foot_force(self, foot_force, joint_angles, leg_idx):
        """
        Compute joint torques needed to produce a desired foot force.
        
        Uses the relationship: τ = J^T @ F
        
        Args:
            foot_force: Desired force [Fx, Fy, Fz] at foot
            joint_angles: Current joint angles [θ1, θ2, θ3]
            leg_idx: Leg index
            
        Returns:
            Joint torques [τ1, τ2, τ3]
        """
        J = self.compute_leg_jacobian(joint_angles, leg_idx)
        tau = J.T @ foot_force
        return tau
    
    def force_error_to_position_shift(self, force_error):
        """
        Convert force error to foot position shift.
        
        The intuition: to increase the force a foot can apply in a direction,
        we shift the foot position in the opposite direction (increases moment arm).
        
        Args:
            force_error: [Fx_error, Fy_error, Fz_error]
            
        Returns:
            Position shift [dx, dy] (Z shift not used for ground contact)
        """
        # Negative gain: to increase force capability, move opposite direction
        k = -self.force_to_position_gain
        
        dx = k * force_error[0]
        dy = k * force_error[1]
        
        return np.array([dx, dy])
    
    def compute_balance_correction(self, roll_rad, pitch_rad, 
                                    roll_rate=0, pitch_rate=0,
                                    roll_error=0, pitch_error=0):
        """
        Main function: compute foot position shifts for balance correction.
        
        This is the entry point for the torque-based balance control.
        
        Args:
            roll_rad: Current roll angle (radians)
            pitch_rad: Current pitch angle (radians)
            roll_rate: Current roll angular velocity (rad/s)
            pitch_rate: Current pitch angular velocity (rad/s)
            roll_error: Roll angle error (desired - actual) in radians
            pitch_error: Pitch angle error (desired - actual) in radians
            
        Returns:
            Dict with 'front' and 'rear' position shifts [dx, dy]
        """
        # Compute desired angular accelerations from errors using PD control
        # α_desired = Kp * θ_error - Kd * θ_rate
        desired_roll_accel = self.kp_roll * roll_error - self.kd_roll * roll_rate
        desired_pitch_accel = self.kp_pitch * pitch_error - self.kd_pitch * pitch_rate
        
        # Compute required ground reaction forces
        desired_forces = self.compute_required_ground_reaction_forces(
            roll_rad, pitch_rad,
            roll_rate, pitch_rate,
            desired_roll_accel, desired_pitch_accel
        )
        
        # Map desired horizontal forces to position shifts
        F_front = desired_forces[self.front_leg]
        F_rear = desired_forces[self.rear_leg]
        
        shift_front = self.force_error_to_position_shift(F_front)
        shift_rear = self.force_error_to_position_shift(F_rear)
        
        # Apply direction inversions
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1
        
        shift_front[0] *= x_dir
        shift_front[1] *= y_dir
        shift_rear[0] *= x_dir
        shift_rear[1] *= y_dir
        
        # Clamp to limits
        shift_front[0] = np.clip(shift_front[0], -self.max_x_shift, self.max_x_shift)
        shift_front[1] = np.clip(shift_front[1], -self.max_y_shift, self.max_y_shift)
        shift_rear[0] = np.clip(shift_rear[0], -self.max_x_shift, self.max_x_shift)
        shift_rear[1] = np.clip(shift_rear[1], -self.max_y_shift, self.max_y_shift)
        
        self.position_shifts = {
            'front': shift_front,
            'rear': shift_rear,
        }
        
        return self.position_shifts
    
    def apply_shifts_to_legslocation(self, legslocation, shifts=None):
        """
        Apply computed shifts to leg locations.
        
        Args:
            legslocation: Current leg positions [[x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]]
            shifts: Optional dict with 'front' and 'rear' [dx, dy] shifts.
                    If None, uses self.position_shifts
            
        Returns:
            Modified legslocation
        """
        if shifts is None:
            shifts = self.position_shifts
            
        # Apply to front stance leg
        legslocation[0][self.front_leg] = (
            self.DEFAULT_STANCE[self.front_leg][0] + shifts['front'][0]
        )
        legslocation[1][self.front_leg] = (
            self.DEFAULT_STANCE[self.front_leg][1] + shifts['front'][1]
        )
        
        # Apply to rear stance leg
        legslocation[0][self.rear_leg] = (
            self.DEFAULT_STANCE[self.rear_leg][0] + shifts['rear'][0]
        )
        legslocation[1][self.rear_leg] = (
            self.DEFAULT_STANCE[self.rear_leg][1] + shifts['rear'][1]
        )
        
        return legslocation
    
    def get_debug_info(self):
        """Return debug information about current state."""
        return {
            'desired_forces': self.desired_forces,
            'actual_torques': self.actual_torques,
            'required_torques': self.required_torques,
            'position_shifts': self.position_shifts,
            'front_leg': self.front_leg,
            'rear_leg': self.rear_leg,
        }


def angle_converter_roll(ax, ay, az):
    """Convert accelerometer readings to roll angle in degrees."""
    return math.degrees(math.atan(ax / math.sqrt(ay * ay + az * az)))

def angle_converter_pitch(ax, ay, az):
    """Convert accelerometer readings to pitch angle in degrees."""
    return math.degrees(math.atan(ay / math.sqrt(ax * ax + az * az)))

def main(use_imu=False, diagonal='FR_BL', lift_height=0.03, invert_x=False, invert_y=False,
         use_torque_control=False):
    """Main control loop for the quadruped robot with diagonal balance.
    
    Args:
        use_imu: Boolean flag to enable/disable IMU usage
        diagonal: Which diagonal to lift ('FR_BL' or 'FL_BR')
        lift_height: Height to lift the diagonal legs (meters)
        invert_x: If True, flip the X shift direction (front/back)
        invert_y: If True, flip the Y shift direction (left/right)
        use_torque_control: If True, use torque-based balance instead of simple gains
    """
    global DIAGONAL_PAIR, LIFT_HEIGHT
    DIAGONAL_PAIR = diagonal
    LIFT_HEIGHT = lift_height
    
    print(f"Starting LQR Diagonal Balance Control")
    print(f"  Diagonal to lift: {diagonal}")
    print(f"  Lift height: {lift_height}m")
    print(f"  Control mode: {'Torque-based' if use_torque_control else 'Gain-based'}")
    print(f"  Invert X: {invert_x}, Invert Y: {invert_y}")
    print(f"  Press Ctrl+C to exit")
    print("-" * 40)
    
    # Parameter configuration
    angle_pitch = []
    angle_roll = []
    
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
    last_print_time = time.time()
    last_print_time2 = time.time()

    command = Command()
    command.pseudo_dance_event = True

    # Measure original roll and pitch angle
    original = esp32.imu_get_data()
    original_roll =  angle_converter_roll(original['ax'], original['ay'], original['az'])
    original_pitch = angle_converter_pitch(original['ax'], original['ay'], original['az'])
    print(f"original_roll & original_pitch: {original_roll}, {original_pitch}")
    original_roll_rad = math.radians(original_roll)
    original_pitch_rad = math.radians(original_pitch)   

    # Create PID controller 
    # Parameters can be tuned to be suitable with the need
    kp = 0.8
    ki = 0.01
    kd = 0.01
    pid_roll = PIDController(kp, ki, kd, original_roll)
    pid_pitch = PIDController(kp, ki, kd, original_pitch)


    # Low-pass filter initializer
    # Parameters can be tuned to be suitable with the need
    sample_rate = 1 / 0.005  # Hz
    cutoff_freq = 6   # Hz
    ax_filtered_lowpass = IIRLowPassFilter(cutoff_freq, sample_rate, order=1)
    ay_filtered_lowpass = IIRLowPassFilter(cutoff_freq, sample_rate, order=1)

    # Kalman filter initializer
    # Parameters can be tuned to be suitable with the need
    initial_state = np.array([0, 0, 0, 0])  # [ax, ay, vx, vy]
    initial_covariance = np.eye(4) * 0.1
    process_noise = np.eye(4) * 0.01
    measurement_noise = np.eye(2) * 0.1

    ekf = RealTimeEKF(initial_state, initial_covariance, process_noise, measurement_noise)
    ekf.start()  # Start the filtering thread

    # LQR setup
    dt = 0.001
    I_r = 1.0
    I_p = 1.0
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    B = np.array([[0, 0],
                  [0, 0],
                  [dt/I_r, 0],
                  [0, dt/I_p]])
    Q = np.diag([1500, 1500, 1, 1])   # penalize roll/pitch strongly
    R = np.diag([0.5, 0.5])       # penalize control effort
    lqr = LQRController(A, B, Q, R)
    
    # Balance controller selection
    if use_torque_control:
        # Torque-based balance controller
        balance_controller = TorqueBasedBalanceController(
            diagonal=diagonal,
            max_x_shift=0.025,
            max_y_shift=0.02,
            force_to_position_gain=0.0005,  # Tune this gain
            kp_roll=50.0,
            kp_pitch=50.0,
            kd_roll=5.0,
            kd_pitch=5.0,
            invert_x=invert_x,
            invert_y=invert_y
        )
        print("Using TorqueBasedBalanceController")
    else:
        # Simple gain-based foot shifter (original)
        balance_controller = None
        
    # Stance foot shifter for balance correction via foot position (always create for fallback)
    # Gains can be tuned: higher = more aggressive foot shifting
    foot_shifter = StanceFootShifter(
        diagonal=diagonal,
        pitch_to_x_gain=0.002,   # 2mm per degree of pitch error
        roll_to_y_gain=0.002,    # 2mm per degree of roll error
        max_x_shift=0.025,       # max 25mm X shift
        max_y_shift=0.02,        # max 20mm Y shift
        invert_x=invert_x,
        invert_y=invert_y
    )

    # Other Variable Initializer
    previous_time = time.time()
    error_roll = 0
    error_pitch = 0
    error_roll_lqr = 0
    error_pitch_lqr = 0
    prev_roll = 0
    prev_pitch = 0

    # Main control loop
    while True:
        now = time.time()
        if now - last_loop < 0.005:  # config.dt (200Hz control loop)
            continue
        last_loop = time.time()

        # Read imu data
        quat_orientation = (
            imu.read_orientation() if use_imu else np.array([1, 0, 0, 0])
        )
        state.quat_orientation = quat_orientation
        
        # IMU data processing
        imu_data = esp32.imu_get_data()
        roll_angle = angle_converter_roll(imu_data['ax'], imu_data['ay'], imu_data['az']) #degree
        pitch_angle = angle_converter_pitch(imu_data['ax'], imu_data['ay'], imu_data['az']) #degree

        # Sensor fusion pipeline
        ekf.add_measurement(ax_filtered_lowpass.update(roll_angle), 
                           ay_filtered_lowpass.update(pitch_angle)) #update ekf data 
        filtered = ekf.get_filtered_output() #get filtered angle data
        filtered_roll_angle = filtered[0]
        filtered_pitch_angle = filtered[1]
        
        # Convert to radians and subtract equilibrium (original posture)
        roll_rad  = math.radians(filtered_roll_angle  - original_roll)
        pitch_rad = math.radians(filtered_pitch_angle - original_pitch)

        # Approximate angular velocities
        dt = now - previous_time
        roll_rate = (roll_rad - prev_roll) / dt
        pitch_rate = (pitch_rad - prev_pitch) / dt
        
        prev_roll, prev_pitch = roll_rad, pitch_rad
        previous_time = now


        # Balance correction logic
        if (abs(filtered_roll_angle) > 1 or abs(filtered_pitch_angle) > 1): #Threshold for the change of roll and pitch angle 
            if movementCtl.movement_now_number >= lib_length - 1 and movementCtl.tick >= movementCtl.now_ticks: #Finish the previous movement
                #previous_time = time.time()
                elapsed_dt = time.time() - previous_time

                # Approximate angular velocities
                #dt = now - previous_time
                roll_rate = (roll_rad - prev_roll) / elapsed_dt
                pitch_rate = (pitch_rad - prev_pitch) / elapsed_dt
                
                prev_roll, prev_pitch = roll_rad, pitch_rad
                previous_time = now

                # Build state vector for LQR
                state_vector = np.array([roll_rad, pitch_rad, roll_rate, pitch_rate])
                # Compute LQR correction
                error_roll_lqr, error_pitch_lqr = lqr.compute(state_vector)

                #Calculate driven angles by PID controller
                error_roll = pid_roll.compute(filtered_roll_angle, elapsed_dt)
                error_pitch = pid_pitch.compute(-filtered_pitch_angle, elapsed_dt)
                previous_time = time.time()
                
                # Compute stance foot position shifts based on tipping
                # Use filtered angles relative to original (equilibrium) position
                pitch_error_deg = filtered_pitch_angle - original_pitch  # positive = forward tipping
                roll_error_deg = filtered_roll_angle - original_roll      # positive = left tipping
                
                if use_torque_control and balance_controller is not None:
                    # Torque-based: compute shifts from force/torque analysis
                    roll_error_rad = -roll_rad  # Error = desired (0) - actual
                    pitch_error_rad = -pitch_rad
                    
                    balance_controller.compute_balance_correction(
                        roll_rad, pitch_rad,
                        roll_rate, pitch_rate,
                        roll_error_rad, pitch_error_rad
                    )
                else:
                    # Simple gain-based
                    foot_shifter.compute_shifts(pitch_error_deg, roll_error_deg)

                
                #Create the movement
                add_movement(error_roll_lqr, -error_pitch_lqr)
                #add_movement(error_roll, error_pitch)
        
        if time.time() - last_print_time >= 2.0:
            print(f"Roll angle & pitch_angle: {roll_angle:.2f}, {pitch_angle:.2f}")
            print(f"Filtered_roll_angle & Filtered_pitch_angle: {filtered_roll_angle:.2f}, {filtered_pitch_angle:.2f}")
            print(f"PID: error_roll & error_pitch: {error_roll:.3f}, {error_pitch:.3f}")
            print(f"LQR: error_roll & error_pitch: {error_roll_lqr:.3f}, {error_pitch_lqr:.3f}")
            
            if use_torque_control and balance_controller is not None:
                debug = balance_controller.get_debug_info()
                print(f"Torque control - Position shifts:")
                print(f"  Front: X={debug['position_shifts']['front'][0]*1000:.1f}mm, "
                      f"Y={debug['position_shifts']['front'][1]*1000:.1f}mm")
                print(f"  Rear:  X={debug['position_shifts']['rear'][0]*1000:.1f}mm, "
                      f"Y={debug['position_shifts']['rear'][1]*1000:.1f}mm")
                print(f"  Required torques: roll={debug['required_torques'][0]:.4f}, "
                      f"pitch={debug['required_torques'][1]:.4f}")
            else:
                shift_info = foot_shifter.get_shift_info()
                print(f"Foot shifts: X_front={shift_info['x_shift_front']*1000:.1f}mm, "
                      f"X_rear={shift_info['x_shift_rear']*1000:.1f}mm, Y={shift_info['y_shift']*1000:.1f}mm")
            print()
            last_print_time = time.time()

        # Movement control
        movementCtl.runMovementScheme()
        command.legslocation = movementCtl.getMovemenLegsLocation()
        
        # Apply foot position shifts to stance legs for balance correction
        if use_torque_control and balance_controller is not None:
            command.legslocation = balance_controller.apply_shifts_to_legslocation(command.legslocation)
        else:
            command.legslocation = foot_shifter.apply_shifts_to_legslocation(command.legslocation)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LQR Diagonal Balance Control for Mini Pupper 2')
    parser.add_argument('--diagonal', type=str, default='FR_BL', choices=['FR_BL', 'FL_BR'],
                       help='Which diagonal to lift: FR_BL (Front-Right + Back-Left) or FL_BR (Front-Left + Back-Right)')
    parser.add_argument('--lift-height', type=float, default=0.03,
                       help='Height to lift diagonal legs in meters (default: 0.03)')
    parser.add_argument('--use-imu', action='store_true',
                       help='Enable external IMU')
    parser.add_argument('--invert-x', action='store_true',
                       help='Flip the X (front/back) shift direction for stance foot corrections')
    parser.add_argument('--invert-y', action='store_true',
                       help='Flip the Y (left/right) shift direction for stance foot corrections')
    parser.add_argument('--torque-control', action='store_true',
                       help='Use torque-based balance control instead of simple gains')
    
    args = parser.parse_args()
    
    main(use_imu=args.use_imu, diagonal=args.diagonal, lift_height=args.lift_height,
         invert_x=args.invert_x, invert_y=args.invert_y, use_torque_control=args.torque_control)