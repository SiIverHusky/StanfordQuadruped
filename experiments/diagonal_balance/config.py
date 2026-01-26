"""
Configuration for Diagonal Balance Experiment.

All tunable parameters are centralized here for easy adjustment.
"""

import numpy as np

# =============================================================================
# DIAGONAL PAIR SELECTION
# =============================================================================
# Which diagonal pair to lift:
#   "FR_BL" - Front-Right + Back-Left (legs 0 and 3)
#   "FL_BR" - Front-Left + Back-Right (legs 1 and 2)
DIAGONAL_PAIR = "FR_BL"

# Leg indices for each diagonal pair
# Leg order: 0=FR, 1=FL, 2=BR, 3=BL
DIAGONAL_PAIRS = {
    "FR_BL": {"lift": [0, 3], "support": [1, 2]},  # Lift FR and BL
    "FL_BR": {"lift": [1, 2], "support": [0, 3]},  # Lift FL and BR
}

# =============================================================================
# MOVEMENT PARAMETERS
# =============================================================================
# Height to lift the diagonal legs (meters)
# Start conservative, max is legliftcap = 0.06
LIFT_HEIGHT = 0.02

# Time to transition to lifted position (seconds) - for single-step lift
LIFT_TIME = 0.5

# Time to hold the diagonal balance (seconds)
HOLD_TIME = 2.0

# Time to return to standing (seconds)
RETURN_TIME = 0.5

# Execution interval (seconds) - matches original dt
DT = 0.015

# =============================================================================
# GRADUAL LIFT PARAMETERS (RECOMMENDED)
# =============================================================================
# Gradual lifting is more stable because the balance controller
# can adjust throughout the lift, rather than trying to recover
# after a sudden movement.

# Number of incremental steps to reach full lift height
GRADUAL_LIFT_STEPS = 10

# Time for each lift step (seconds) - balance loop runs between steps
GRADUAL_LIFT_STEP_TIME = 0.3

# =============================================================================
# CENTER OF MASS SHIFT
# =============================================================================
# Enable gradual CoM shift before lifting
# This shifts the body weight over the supporting diagonal for stability
ENABLE_COM_SHIFT = True

# CoM shift distance in X direction (meters) - forward/backward
# Positive = shift toward front-left/back-right diagonal
COM_SHIFT_X = 0.02

# CoM shift distance in Y direction (meters) - left/right
COM_SHIFT_Y = 0.01

# Time for CoM shift transition (seconds)
COM_SHIFT_TIME = 0.5

# Body lean during diagonal lift (degrees)
# This tilts the body toward the support legs to keep CoM centered
BODY_LEAN_ROLL = 8.0   # Roll toward support side
BODY_LEAN_PITCH = 5.0  # Pitch toward support side

# =============================================================================
# DEFAULT STANCE (from MovementGroup.py)
# =============================================================================
# [[[x, y, z]], ...] for each leg in body frame
# Leg order: FR, FL, BR, BL
DEFAULT_STANCE = [
    [[0.06, -0.05, -0.07]],   # Leg 0: Front-Right
    [[0.06, 0.05, -0.07]],    # Leg 1: Front-Left
    [[-0.06, -0.05, -0.07]],  # Leg 2: Back-Right
    [[-0.06, 0.05, -0.07]],   # Leg 3: Back-Left
]

# Stance as 3x4 matrix for kinematics (x, y, z rows; FR, FL, BR, BL columns)
DEFAULT_STANCE_MATRIX = np.array([
    [0.06, 0.06, -0.06, -0.06],   # X
    [-0.05, 0.05, -0.05, 0.05],   # Y
    [-0.07, -0.07, -0.07, -0.07], # Z
])

# =============================================================================
# PID CONTROLLER
# =============================================================================
PID_KP = 0.8    # Proportional gain
PID_KI = 0.01   # Integral gain (keep small to avoid windup)
PID_KD = 0.01   # Derivative gain (keep small to avoid noise amplification)

# PID output limits (degrees) - these get converted to meters for foot shift
PID_OUTPUT_MIN = -15.0
PID_OUTPUT_MAX = 15.0

# =============================================================================
# FOOT POSITION BALANCE CORRECTION
# =============================================================================
# Instead of body attitude, we shift the support leg foot positions
# to move the center of pressure and counteract tipping.

# Conversion factor: degrees of tilt error -> meters of foot shift
# A 10° tilt error should produce ~0.01m (1cm) of foot shift
# So 1° = 0.001m = 1mm
TILT_TO_FOOT_SHIFT_GAIN = 0.001  # meters per degree

# Maximum foot shift from nominal position (meters)
MAX_FOOT_SHIFT_X = 0.03  # Forward/backward
MAX_FOOT_SHIFT_Y = 0.02  # Left/right

# Direction mapping for foot shifts:
# - Positive pitch (nose down) -> shift feet forward (negative X correction)
# - Positive roll (right side down) -> shift feet right (negative Y correction)
PITCH_TO_X_SIGN = -1.0  # Pitch error -> X foot shift direction
ROLL_TO_Y_SIGN = -1.0   # Roll error -> Y foot shift direction

# Integral windup limits
PID_INTEGRAL_MIN = -10.0
PID_INTEGRAL_MAX = 10.0

# =============================================================================
# FILTER PARAMETERS
# =============================================================================
# IIR Low-pass filter
FILTER_SAMPLE_RATE = 1.0 / 0.005  # Hz (200 Hz control loop)
FILTER_CUTOFF_FREQ = 6.0          # Hz

# Filter order (1 or 2)
FILTER_ORDER = 1

# EKF parameters
EKF_INITIAL_STATE = np.array([0, 0, 0, 0])  # [ax, ay, vx, vy]
EKF_INITIAL_COVARIANCE = np.eye(4) * 0.1
EKF_PROCESS_NOISE = np.eye(4) * 0.01
EKF_MEASUREMENT_NOISE = np.eye(2) * 0.1

# =============================================================================
# SAFETY THRESHOLDS
# =============================================================================
# Maximum tilt before abort (degrees)
# Note: During diagonal lift, expect 10-20° of tilt. Set higher for testing.
MAX_TILT_ABORT = 30.0

# Minimum tilt to trigger balance correction (degrees)
TILT_THRESHOLD = 1.0

# Number of consecutive stable cycles before lifting
SUSTAINED_TRIGGER_CYCLES = 10

# Maximum time in diagonal balance before forced return to stand (seconds)
TIMEOUT_S = 5.0

# Control loop target period (seconds)
CONTROL_LOOP_DT = 0.005

# =============================================================================
# JOINT LIMITS (radians)
# =============================================================================
# These limits are based on what the Mini Pupper 2 hardware actually supports.
# The existing MovementGroup.py uses these stance positions successfully.
JOINT_LIMITS = {
    "abduction": {"min": -0.6, "max": 0.6},     # ~±34 degrees
    "hip": {"min": -1.0, "max": 1.85},          # ~-57 to +106 degrees
    "knee": {"min": -2.8, "max": 0.5},          # ~-160 to +28 degrees
}

# =============================================================================
# PWM LIMITS
# =============================================================================
PWM_MIN = 500   # microseconds
PWM_MAX = 2500  # microseconds
PWM_NEUTRAL = 1500

# =============================================================================
# MODES
# =============================================================================
class Mode:
    """Operating modes for the diagonal balancer."""
    VALIDATOR = "validator"  # Dry-run, no hardware
    BENCH = "bench"          # Hardware, robot secured
    PHYSICAL = "physical"    # Full operation

# =============================================================================
# DEBUG
# =============================================================================
DEBUG_PRINT = True  # Print debug info during operation
LOG_TO_CSV = False  # Log sensor data to CSV file
LOG_FILE = "diagonal_balance_log.csv"
