# Diagonal Balance Experiment for Mini Pupper 2

Balance the Mini Pupper 2 on two diagonal legs (FR+BL or FL+BR) while the other two legs are lifted, similar to Unitree Go2 and Boston Dynamics Spot demos.

## Overview

This experiment is isolated from the main StanfordQuadruped codebase. It reuses existing modules via imports but does not modify any original files.

## Files

| File | Description |
|------|-------------|
| `config.py` | All tunable parameters, safety thresholds, and constants |
| `diagonal_movement.py` | Generates Movement sequences for diagonal leg lift with CoM shift |
| `filters.py` | IIR low-pass filter and Extended Kalman Filter for sensor fusion |
| `safety.py` | Safety state machine with abort conditions and emergency stop |
| `diagonal_balancer.py` | Main control loop that orchestrates the balance |
| `validator.py` | Dry-run tool to validate kinematics and joint limits |

## Usage

### Step 1: Validate (no hardware needed)
```bash
cd ~/StanfordQuadruped
python -m experiments.diagonal_balance.validator
```
This runs the diagonal lift movements through inverse kinematics and checks that joint angles and PWM values are within safe ranges.

### Step 2: Bench Test (robot secured, servos powered)
```bash
python -m experiments.diagonal_balance.diagonal_balancer --mode bench --pair FR_BL
```
With the robot secured (e.g., on a stand), this will execute a gradual diagonal lift and return to stand.

### Step 3: Physical Test
```bash
python -m experiments.diagonal_balance.diagonal_balancer --mode physical --pair FR_BL
```
Full balance loop with IMU feedback. Start with conservative parameters.

---

## Configuration Reference

All parameters are in `config.py`. Edit and re-sync to the robot to apply changes.

### Diagonal Pair Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIAGONAL_PAIR` | `"FR_BL"` | Which diagonal to lift: `"FR_BL"` (Front-Right + Back-Left) or `"FL_BR"` (Front-Left + Back-Right) |

### Movement Timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LIFT_HEIGHT` | `0.02` | Height to lift legs (meters). Start low (0.02), max 0.06. Higher = harder to balance |
| `LIFT_TIME` | `0.5` | Time for single-step lift (seconds). Only used if gradual lift disabled |
| `HOLD_TIME` | `2.0` | Time to hold diagonal balance at full height (seconds) |
| `RETURN_TIME` | `0.5` | Time to return to standing (seconds) |
| `DT` | `0.015` | Movement execution interval (seconds). Lower = smoother but more CPU |

### Gradual Lift (RECOMMENDED for stability)

The gradual lift raises legs in small increments, allowing the balance controller to adjust throughout the entire lift. This is much more stable than a single sudden lift.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRADUAL_LIFT_STEPS` | `10` | Number of incremental steps to reach full height. More steps = slower but more stable |
| `GRADUAL_LIFT_STEP_TIME` | `0.3` | Time per step (seconds). Total lift time = steps × step_time (default: 3s total) |

**If legs move too fast:** Increase `GRADUAL_LIFT_STEPS` or `GRADUAL_LIFT_STEP_TIME`

### Center of Mass Shift

Before lifting, the robot shifts its weight toward the support legs for better initial stability.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_COM_SHIFT` | `True` | Enable/disable the CoM shift phase |
| `COM_SHIFT_X` | `0.02` | Forward/backward shift (meters) |
| `COM_SHIFT_Y` | `0.01` | Left/right shift (meters) |
| `COM_SHIFT_TIME` | `0.5` | Time for CoM shift transition (seconds) |
| `BODY_LEAN_ROLL` | `8.0` | Body roll toward support legs (degrees) |
| `BODY_LEAN_PITCH` | `5.0` | Body pitch toward support legs (degrees) |

### PID Controller

Controls how aggressively the robot corrects for tilt during balance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PID_KP` | `0.8` | **Proportional gain** - Main correction strength. Higher = faster response but can oscillate |
| `PID_KI` | `0.01` | **Integral gain** - Corrects persistent errors. Keep small to avoid windup |
| `PID_KD` | `0.01` | **Derivative gain** - Dampens oscillations. Keep small to avoid noise amplification |
| `PID_OUTPUT_MIN` | `-15.0` | Minimum PID output (degrees) |
| `PID_OUTPUT_MAX` | `15.0` | Maximum PID output (degrees) |
| `PID_INTEGRAL_MIN` | `-10.0` | Integral anti-windup lower limit |
| `PID_INTEGRAL_MAX` | `10.0` | Integral anti-windup upper limit |

**Tuning tips:**
- **Oscillating/jerky?** Reduce `PID_KP` (try 0.5, 0.3, 0.2)
- **Slow to respond?** Increase `PID_KP` slightly (try 1.0, 1.2)
- **Drifting to one side?** Small increase to `PID_KI` (try 0.02, 0.05)
- **Overcorrecting after perturbation?** Increase `PID_KD` slightly (try 0.02, 0.05)

### Foot Position Balance Correction

Instead of adjusting body attitude (which doesn't work well with 2 legs lifted), the balance controller shifts the support leg foot positions to move the center of pressure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TILT_TO_FOOT_SHIFT_GAIN` | `0.001` | Meters of foot shift per degree of tilt error. 1° tilt = 1mm shift |
| `MAX_FOOT_SHIFT_X` | `0.03` | Maximum forward/backward foot shift (meters). 3cm limit |
| `MAX_FOOT_SHIFT_Y` | `0.02` | Maximum left/right foot shift (meters). 2cm limit |
| `PITCH_TO_X_SIGN` | `-1.0` | Direction mapping: positive pitch → negative X shift (forward) |
| `ROLL_TO_Y_SIGN` | `-1.0` | Direction mapping: positive roll → negative Y shift (right) |

**If corrections seem backwards:** Flip the sign (change -1.0 to 1.0 or vice versa)

**If corrections are too aggressive:** Reduce `TILT_TO_FOOT_SHIFT_GAIN` (try 0.0005)

**If corrections are too weak:** Increase `TILT_TO_FOOT_SHIFT_GAIN` (try 0.002, 0.003)

### Sensor Filtering

Filters smooth out IMU noise before feeding to the PID controller.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FILTER_SAMPLE_RATE` | `200` | Sampling frequency in Hz (1/0.005) |
| `FILTER_CUTOFF_FREQ` | `6.0` | Low-pass filter cutoff (Hz). Lower = smoother but more lag |
| `FILTER_ORDER` | `1` | Filter order (1 or 2). Higher = sharper cutoff |

**If balance response is laggy:** Increase `FILTER_CUTOFF_FREQ` (try 10, 15)

**If balance is noisy/jittery:** Decrease `FILTER_CUTOFF_FREQ` (try 4, 3)

### Safety Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_TILT_ABORT` | `30.0` | Emergency stop if roll or pitch exceeds this (degrees) |
| `TILT_THRESHOLD` | `1.0` | Minimum tilt to trigger balance correction (degrees) |
| `TIMEOUT_S` | `5.0` | Maximum balance duration before automatic return to stand (seconds) |
| `CONTROL_LOOP_DT` | `0.005` | Control loop period (seconds). 200Hz. Don't change unless needed |

### Joint Limits

| Parameter | Description |
|-----------|-------------|
| `JOINT_LIMITS["abduction"]` | Hip abduction: ±34° (-0.6 to 0.6 rad) |
| `JOINT_LIMITS["hip"]` | Hip flexion: -57° to +106° (-1.0 to 1.85 rad) |
| `JOINT_LIMITS["knee"]` | Knee: -160° to +28° (-2.8 to 0.5 rad) |

---

## Quick Tuning Guide

### Problem: Legs move too fast / jerky corrections

1. **Reduce PID_KP**: `0.8` → `0.5` → `0.3`
2. **Reduce TILT_TO_FOOT_SHIFT_GAIN**: `0.001` → `0.0007` → `0.0005`
3. **Increase GRADUAL_LIFT_STEP_TIME**: `0.3` → `0.5` → `0.7`

### Problem: Robot tips over before corrections take effect

1. **Increase PID_KP**: `0.8` → `1.0` → `1.5`
2. **Increase TILT_TO_FOOT_SHIFT_GAIN**: `0.001` → `0.002` → `0.003`
3. **Increase BODY_LEAN_ROLL/PITCH**: `8.0/5.0` → `10.0/8.0` → `12.0/10.0`

### Problem: Balance corrections go the wrong direction

1. **Flip PITCH_TO_X_SIGN**: `-1.0` → `1.0`
2. **Flip ROLL_TO_Y_SIGN**: `-1.0` → `1.0`

### Problem: Oscillating back and forth

1. **Reduce PID_KP**: `0.8` → `0.4`
2. **Increase PID_KD**: `0.01` → `0.03` → `0.05`
3. **Reduce FILTER_CUTOFF_FREQ**: `6.0` → `4.0` (adds damping via filtering)

### Problem: Drifts slowly to one side

1. **Increase PID_KI**: `0.01` → `0.02` → `0.05`
2. **Check IMU calibration baseline**

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ESP32 IMU     │────▶│   filters.py    │────▶│  diagonal_      │
│ (ax, ay, az)    │     │ IIR + EKF       │     │  balancer.py    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │ diagonal_       │◀─────────────┘
                        │ movement.py     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ MovementScheme  │  (from src/)
                        │ Controller      │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ HardwareInterface│
                        │ Kinematics      │
                        └─────────────────┘
```

## Dependencies

- numpy
- transforms3d
- MangDang.mini_pupper (external package)
5. Use `validator.py` after any parameter change to ensure joint limits are respected

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Robot tips over immediately | Reduce `LIFT_HEIGHT`, check CoM shift direction |
| Oscillations | Reduce `PID_KP`, increase filter cutoff |
| No response to tilt | Increase `PID_KP`, check IMU connection |
| Joints at limit | Reduce `LIFT_HEIGHT`, check calibration |

## License

MIT License - see main repository LICENSE file.
