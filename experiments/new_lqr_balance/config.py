"""
Configuration for Reaction Wheel LQR Balance Controller.

Based on the reaction-wheel inverted pendulum model:
- Body + lifted legs = reaction wheel
- Supporting legs = pendulum

Adapted for Mini Pupper 2 physical dimensions.
"""

import numpy as np

# =============================================================================
# CONTROL TIMING
# =============================================================================
CONTROL_DT = 0.005  # 200 Hz control loop (faster for balance)
TIMEOUT_S = 60.0    # Maximum run time

# Movement phase timing (seconds)
LIFT_TIME = 1.0     # Time to lift legs
HOLD_TIME = 5.0     # Time to hold diagonal stance (balance phase)
RETURN_TIME = 1.0   # Time to return to normal stance

# =============================================================================
# DIAGONAL LEG SELECTION
# =============================================================================
# Leg indices: 0=FR, 1=FL, 2=BR, 3=BL
SUPPORT_PAIR = "FL_BR"  # Default: Front-Left + Back-Right support

DIAGONAL_PAIRS = {
    "FR_BL": {"support": [0, 3], "lift": [1, 2]},  # Support FR and BL
    "FL_BR": {"support": [1, 2], "lift": [0, 3]},  # Support FL and BR
}


# =============================================================================
# ROBOT PHYSICAL PARAMETERS (Mini Pupper 2)
# =============================================================================
class RobotParams:
    """Physical parameters of Mini Pupper 2."""
    
    # Masses (kg)
    FRAME_MASS = 0.560
    MODULE_MASS = 0.080  # Per hip module
    LEG_MASS = 0.030     # Per leg
    
    BODY_MASS = FRAME_MASS + 4 * MODULE_MASS
    TOTAL_MASS = FRAME_MASS + 4 * (MODULE_MASS + LEG_MASS)
    
    # For reaction wheel model:
    # - Pendulum mass = support legs + trunk
    # - Reaction wheel mass = lifted legs
    PENDULUM_MASS = BODY_MASS + 2 * LEG_MASS  # Body + 2 support legs
    WHEEL_MASS = 2 * (MODULE_MASS + LEG_MASS)  # 2 lifted leg modules
    
    # Geometry (m)
    LEG_FB = 0.10       # Front-back distance from center to leg
    LEG_LR = 0.04       # Left-right distance from center to leg plane
    LEG_L1 = 0.1235     # Upper leg length
    LEG_L2 = 0.115      # Lower leg length
    LEG_LENGTH = LEG_L1 + LEG_L2  # Total leg length
    
    # Standing height (nominal)
    STANDING_HEIGHT = 0.16
    
    # Center of mass height above hip
    COM_HEIGHT = 0.03
    
    # Distance from hip to CoM of leg
    LEG_COM_DIST = (LEG_L1 + LEG_L2) / 2
    
    # Abduction offset
    ABDUCTION_OFFSET = 0.03
    
    # Effective pendulum length (hip to ground)
    PENDULUM_LENGTH = STANDING_HEIGHT
    
    # Diagonal distance between support feet
    DIAGONAL_DISTANCE = np.sqrt((2 * LEG_FB) ** 2 + (2 * LEG_LR) ** 2)
    
    # Gravity
    GRAVITY = 9.81
    
    # Friction coefficient
    STATIC_FRICTION = 0.4
    
    # Moment of inertia approximations
    @classmethod
    def body_inertia(cls) -> float:
        """Body moment of inertia about hip (kg·m²)."""
        # Approximate body as point mass at COM_HEIGHT
        return cls.BODY_MASS * (cls.COM_HEIGHT ** 2)
    
    @classmethod
    def wheel_inertia(cls) -> float:
        """Reaction wheel (lifted legs) moment of inertia (kg·m²)."""
        # Approximate lifted legs as point masses at leg attachment points
        return cls.WHEEL_MASS * (cls.LEG_FB ** 2)
    
    @classmethod
    def pendulum_inertia(cls) -> float:
        """Total pendulum inertia about foot contact (kg·m²)."""
        # Body inertia about foot
        I_body = cls.BODY_MASS * (cls.PENDULUM_LENGTH ** 2)
        # Legs contribute to pendulum inertia
        I_legs = 2 * cls.LEG_MASS * (cls.PENDULUM_LENGTH / 2) ** 2
        return I_body + I_legs


# =============================================================================
# SENSOR PARAMETERS
# =============================================================================
class SensorParams:
    """Parameters for sensor fusion and filtering."""
    
    # Gyro drift compensation gain (from spec: ω_comp = ω_gyro + Kc * θ_rel)
    GYRO_DRIFT_GAIN = 0.01  # Kc
    
    # Complementary filter weights
    ACCEL_WEIGHT = 0.02    # Trust accelerometer this much
    GYRO_WEIGHT = 0.98     # Trust gyro this much
    
    # Low-pass filter cutoff (Hz)
    ANGLE_LPF_CUTOFF = 5.0
    RATE_LPF_CUTOFF = 10.0
    
    # Encoder noise (radians)
    ENCODER_NOISE_STD = 0.01
    
    # IMU noise
    GYRO_NOISE_STD = 0.01      # rad/s
    ACCEL_NOISE_STD = 0.1      # m/s²


# =============================================================================
# CONTROL PARAMETERS
# =============================================================================
class ControlParams:
    """LQR controller parameters."""
    
    # State vector: X = [θ_rel, θ_body, ω_rel, ω_body]
    # 
    # θ_rel: relative angle between body and supporting legs
    # θ_body: absolute body inclination (tilt from vertical)
    # ω_rel: angular velocity of relative angle
    # ω_body: angular velocity of body inclination
    
    # State cost matrix Q - penalize deviations from equilibrium
    Q = np.diag([
        50.0,    # θ_rel - relative angle (moderate penalty)
        200.0,   # θ_body - body tilt (high penalty - primary stabilization)
        5.0,     # ω_rel - relative angular velocity
        20.0,    # ω_body - body angular velocity
    ])
    
    # Control cost matrix R - penalize torque usage
    R = np.diag([
        1.0,     # Base hip torque
    ])
    
    # Cooperative torque distribution gains
    # u_c = Kf*(θ_front − θ_rel) + Kr*(θ_rear − θ_rel)
    KF_COOPERATIVE = 5.0   # Front leg cooperative gain
    KR_COOPERATIVE = 5.0   # Rear leg cooperative gain
    
    # Maximum control torque (Nm)
    MAX_TORQUE = 0.5
    
    # Maximum angular velocity for state estimation
    MAX_ANGULAR_VELOCITY = 5.0  # rad/s


# =============================================================================
# SAFETY PARAMETERS
# =============================================================================
class SafetyParams:
    """Safety limits and thresholds."""
    
    # Maximum body tilt before triggering safe-stand (radians)
    MAX_BODY_TILT = np.radians(30.0)  # 30 degrees
    
    # Maximum relative angle (radians)
    MAX_RELATIVE_ANGLE = np.radians(20.0)  # 20 degrees
    
    # Maximum angular velocity before safety cutoff
    MAX_ANGULAR_RATE = 3.0  # rad/s
    
    # Torque saturation with anti-windup
    TORQUE_LIMIT = 0.5  # Nm
    
    # Anti-windup integrator limit
    INTEGRATOR_LIMIT = 0.3  # Nm
    
    # Fall detection - trigger safe-stand if exceeded
    FALL_TILT_THRESHOLD = np.radians(35.0)
    
    # Minimum time between safety resets (seconds)
    SAFETY_RESET_COOLDOWN = 2.0


# =============================================================================
# DEFAULT STANCE (crouched position for balance testing)
# =============================================================================
def get_default_stance() -> np.ndarray:
    """
    Get the default crouched stance matrix for balance testing.
    
    Uses a lower, more stable stance similar to lqr_diagonal_balance.
    
    Returns:
        3x4 matrix of foot positions [x, y, z] for each leg [FR, FL, BR, BL]
    """
    # Smaller footprint for crouched stance
    delta_x = 0.06  # Front/back spread (was 0.10)
    delta_y = 0.05  # Left/right spread (was 0.09)
    z_ref = -0.07   # Crouched height (was -0.16)
    
    return np.array([
        [delta_x, delta_x, -delta_x, -delta_x],   # X positions
        [-delta_y, delta_y, -delta_y, delta_y],   # Y positions
        [z_ref, z_ref, z_ref, z_ref],              # Z positions
    ])


DEFAULT_STANCE = get_default_stance()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_TO_FILE = True
LOG_DIR = "logs"
DEBUG_PRINT = True
DEBUG_INTERVAL = 0.5  # Print debug info every 0.5 seconds
