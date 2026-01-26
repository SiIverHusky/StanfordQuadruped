"""
Configuration for LQR Diagonal Balance Experiment.

Parameters are based on the paper:
"A New Balance Control Approach for Quadruped Robot with Diagonal Leg Support"

Adapted for Mini Pupper 2 physical dimensions.
"""

import numpy as np

# =============================================================================
# ROBOT PHYSICAL PARAMETERS (Mini Pupper 2)
# =============================================================================
# These are adapted from the paper's Table 1 to match Mini Pupper 2

class RobotParams:
    """Physical parameters of Mini Pupper 2."""
    
    # Masses (kg) - from pupper/Config.py
    FRAME_MASS = 0.560
    MODULE_MASS = 0.080  # Per hip module
    LEG_MASS = 0.030     # Per leg
    BODY_MASS = FRAME_MASS + 4 * MODULE_MASS  # Trunk mass
    TOTAL_MASS = FRAME_MASS + 4 * (MODULE_MASS + LEG_MASS)
    
    # Geometry (m) - from pupper/Config.py
    LEG_FB = 0.10       # Front-back distance from center to leg
    LEG_LR = 0.04       # Left-right distance from center to leg plane
    LEG_L1 = 0.1235     # Upper leg length
    LEG_L2 = 0.115      # Lower leg length
    LEG_LENGTH = LEG_L1 + LEG_L2  # Total leg length (extended)
    
    # Computed from geometry
    HALF_WIDTH = LEG_LR + 0.03  # W in paper (half trunk width)
    DIAGONAL_DISTANCE = np.sqrt((2 * LEG_FB) ** 2 + (2 * LEG_LR) ** 2)  # D in paper
    
    # Abduction offset
    ABDUCTION_OFFSET = 0.03
    
    # Gravity
    GRAVITY = 9.81
    
    # Friction coefficients
    STATIC_FRICTION = 0.4
    KINETIC_FRICTION = 0.3
    
    # Standing height (nominal leg extension)
    STANDING_HEIGHT = 0.16  # From default_z_ref


# =============================================================================
# DIAGONAL PAIR SELECTION
# =============================================================================
# Which diagonal pair to use for support:
#   "FR_BL" - Front-Right + Back-Left support (lift FL + BR)
#   "FL_BR" - Front-Left + Back-Right support (lift FR + BL)
SUPPORT_PAIR = "FL_BR"

# Leg indices: 0=FR, 1=FL, 2=BR, 3=BL
DIAGONAL_PAIRS = {
    "FR_BL": {"support": [0, 3], "lift": [1, 2]},  # Support FR and BL
    "FL_BR": {"support": [1, 2], "lift": [0, 3]},  # Support FL and BR
}


# =============================================================================
# COORDINATE SYSTEM
# =============================================================================
# Z-Y-K angle representation as described in Section 3 of the paper.
# K-axis is along the diagonal line connecting the two supporting legs.

def compute_diagonal_angle():
    """Compute the angle of the diagonal axis relative to X-axis."""
    return np.arctan2(2 * RobotParams.LEG_LR, 2 * RobotParams.LEG_FB)

DIAGONAL_AXIS_ANGLE = compute_diagonal_angle()


# =============================================================================
# LQR CONTROL PARAMETERS
# =============================================================================
class LQRParams:
    """LQR controller tuning parameters."""
    
    # State vector: X = [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]
    # (front leg angle, velocity, hind leg angle, velocity, body angle, velocity)
    
    # State cost matrix Q - penalize angle deviations
    # Higher values = more aggressive correction
    Q = np.diag([
        100.0,   # θ_F - front leg angle
        1.0,     # θ_F_dot - front leg angular velocity
        100.0,   # θ_H - hind leg angle
        1.0,     # θ_H_dot - hind leg angular velocity
        500.0,   # θ_body - body pitch angle (most important)
        10.0,    # θ_body_dot - body angular velocity
    ])
    
    # Control cost matrix R - penalize torque usage
    # Higher values = more conservative control
    R = np.diag([
        0.1,     # τ_F - front hip torque
        0.1,     # τ_H - hind hip torque
    ])
    
    # Torque limits (Nm) - based on servo capabilities
    # Mini Pupper uses small servos, so limits are lower than paper
    MAX_HIP_TORQUE = 0.5  # Nm
    
    # Maximum angular velocity for state estimation
    MAX_ANGULAR_VELOCITY = 5.0  # rad/s


# =============================================================================
# PART B: ALONG-DIAGONAL CONTROL (PID)
# =============================================================================
class PartBParams:
    """PID parameters for along-diagonal disturbance control."""
    
    KP = 50.0    # Proportional gain
    KI = 5.0     # Integral gain
    KD = 2.0     # Derivative gain
    
    # Output limits
    MAX_TORQUE = 0.3  # Nm
    
    # Integral windup limits
    INTEGRAL_MAX = 0.5


# =============================================================================
# PART C: HEIGHT CONTROL (PID)
# =============================================================================
class PartCParams:
    """PID parameters for height maintenance."""
    
    KP = 100.0   # Proportional gain
    KI = 10.0    # Integral gain
    KD = 5.0     # Derivative gain
    
    # Output limits
    MAX_FORCE = 5.0  # N
    
    # Target height
    TARGET_HEIGHT = RobotParams.STANDING_HEIGHT
    
    # Integral windup limits
    INTEGRAL_MAX = 1.0


# =============================================================================
# KALMAN FILTER PARAMETERS
# =============================================================================
class KalmanParams:
    """Kalman filter tuning for noisy sensors."""
    
    # Process noise covariance (model uncertainty)
    # Higher = trust measurements more
    Q_PROCESS = np.diag([
        0.001,   # Angle process noise
        0.01,    # Angular velocity process noise
    ])
    
    # Measurement noise covariance
    # From paper: joint angle sensors std = 0.001 rad, posture std = 0.005 rad
    R_JOINT_ANGLE = 0.001 ** 2  # Variance for joint angle sensors
    R_POSTURE = 0.005 ** 2      # Variance for IMU posture sensors
    
    R_MEASUREMENT = np.diag([
        R_JOINT_ANGLE,  # Front leg angle
        R_JOINT_ANGLE,  # Hind leg angle
        R_POSTURE,      # Body angle
    ])


# =============================================================================
# TIMING PARAMETERS
# =============================================================================
CONTROL_DT = 0.005      # Control loop period (seconds) - 200 Hz
SENSOR_DT = 0.005       # Sensor reading period (seconds)
TIMEOUT_S = 30.0        # Maximum balance time before returning to stand

# Movement timing
LIFT_TIME = 1.0         # Time to lift legs (seconds)
HOLD_TIME = 5.0         # Time to hold diagonal stance (seconds)
RETURN_TIME = 1.0       # Time to return to normal stance (seconds)


# =============================================================================
# SAFETY LIMITS
# =============================================================================
class SafetyLimits:
    """Safety thresholds to prevent damage."""
    
    # Maximum body tilt before emergency stop (radians)
    # During diagonal stance, significant tilt is expected - be very generous
    MAX_BODY_TILT = np.deg2rad(70)  # Allow up to 70° during balance testing
    
    # Maximum leg angle deviation from nominal (radians)
    MAX_LEG_DEVIATION = np.deg2rad(60)
    
    # Minimum ground contact force threshold
    MIN_CONTACT_FORCE = 0.1  # N (estimated from leg position)
    
    # Maximum angular velocity (rad/s)
    MAX_ANGULAR_VELOCITY = 5.0  # From paper: 4 rad/s yaw tested, add margin
    
    # Emergency torque limit
    EMERGENCY_TORQUE_LIMIT = 1.0  # Nm
    
    # Minimum stability margin before emergency stop
    # Set to 0 to effectively disable this check
    MIN_STABILITY_MARGIN = 0.0  # Disabled for testing


# =============================================================================
# JOINT LIMITS (radians) - from original experiment
# =============================================================================
JOINT_LIMITS = {
    "abduction": {"min": -0.6, "max": 0.6},
    "hip": {"min": -1.0, "max": 1.85},
    "knee": {"min": -2.8, "max": 0.5},
}


# =============================================================================
# DEFAULT STANCE (from original config)
# =============================================================================
DEFAULT_STANCE_MATRIX = np.array([
    [0.06, 0.06, -0.06, -0.06],   # X (front/back)
    [-0.05, 0.05, -0.05, 0.05],   # Y (left/right)
    [-0.07, -0.07, -0.07, -0.07], # Z (height)
])


# =============================================================================
# DEBUG AND LOGGING
# =============================================================================
DEBUG_PRINT = True
LOG_TO_FILE = True
LOG_FILE = "lqr_diagonal_balance_log.csv"


# =============================================================================
# OPERATING MODES
# =============================================================================
class Mode:
    """Operating modes."""
    SIMULATION = "simulation"  # Pure simulation, no hardware
    BENCH = "bench"            # Hardware on bench (secured)
    PHYSICAL = "physical"      # Full physical operation
