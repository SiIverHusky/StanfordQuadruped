# Reaction Wheel LQR Balance Controller

## Overview

This module implements a two-leg diagonal balance controller using the **reaction-wheel inverted pendulum** model:

- Robot stands on two diagonal legs; other two legs lifted
- Body + lifted legs → **reaction wheel**
- Supporting legs → **pendulum**

## System Model

### State Variables
```
X = [θ_rel, θ_body, ω_rel, ω_body]^T

where:
  θ_rel  : relative angle between body and supporting legs
  θ_body : absolute body inclination angle
  ω_rel  : angular velocity of θ_rel
  ω_body : angular velocity of θ_body
```

### Control Law
```
u = -k^T X         (LQR state feedback)

Cooperative torque distribution:
  u_c = Kf*(θ_front − θ_rel) + Kr*(θ_rear − θ_rel)
  u_front = (u / 2) + u_c
  u_rear  = (u / 2) − u_c
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ReactionWheelBalancer                       │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   IMU       │───▶│    State     │───▶│   LQR Controller  │  │
│  │  + Encoders │    │  Estimator   │    │                   │  │
│  └─────────────┘    └──────────────┘    └─────────┬─────────┘  │
│                                                   │             │
│                                          base_torque            │
│                                                   │             │
│                     ┌─────────────────────────────▼──────────┐  │
│                     │     Cooperative Distribution           │  │
│                     │   u_front = (u/2) + u_c               │  │
│                     │   u_rear  = (u/2) - u_c               │  │
│                     └─────────────────────────────┬──────────┘  │
│                                                   │             │
│                     ┌─────────────────────────────▼──────────┐  │
│                     │     Torque to Position Converter       │  │
│                     │   (Servo impedance model)              │  │
│                     └─────────────────────────────┬──────────┘  │
│                                                   │             │
│                                              position_cmds      │
│                                                   │             │
│  ┌───────────────┐                   ┌────────────▼──────────┐  │
│  │ Safety Monitor│◀──────────────────│     Hardware I/F     │  │
│  └───────────────┘                   └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. State Estimator (`state_estimator.py`)
- Complementary filter for body angle (gyro + accelerometer)
- Gyro drift compensation: `ω_comp = ω_gyro + Kc * θ_rel`
- Coordinate transform to diagonal axis

### 2. LQR Controller (`lqr_controller.py`)
- Solves algebraic Riccati equation for optimal gains
- Fallback to hand-tuned PD gains if solve fails
- Anti-windup integrator for steady-state errors

### 3. Torque Converter (`torque_converter.py`)
- Maps desired torques to position commands
- Uses servo stiffness model: `τ = Kp*(θ_cmd - θ_actual)`
- Rate limiting for smooth motion

### 4. Safety Monitor (`safety.py`)
- Fall detection with emergency stop
- Torque saturation with anti-windup
- Data logging for tuning

## Usage

### Simulation Mode
```bash
cd experiments/new_lqr_balance
python controller.py --mode simulation --duration 30
```

### Bench Testing (legs hanging)
```bash
python controller.py --mode bench --duration 30
```

### Physical Testing
```bash
python controller.py --mode physical --duration 30
```

### Changing Support Legs
```bash
python controller.py --support FR_BL  # Use Front-Right + Back-Left
python controller.py --support FL_BR  # Use Front-Left + Back-Right (default)
```

## Tuning Guidelines

### 1. Start with Low Gains
The default LQR gains in `config.py` are conservative. Increase gradually:
```python
# In config.py, ControlParams
Q = np.diag([
    50.0,    # θ_rel - start here
    200.0,   # θ_body - primary stabilization
    5.0,     # ω_rel
    20.0,    # ω_body
])
```

### 2. Verify Drift Compensation
Compare gyro-integrated angle vs encoder feedback:
```python
# In config.py, SensorParams
GYRO_DRIFT_GAIN = 0.01  # Increase if drift is high
```

### 3. Tune Cooperative Gains
Suppress yaw twist by adjusting `Kf` and `Kr`:
```python
# In config.py, ControlParams
KF_COOPERATIVE = 5.0   # Front leg
KR_COOPERATIVE = 5.0   # Rear leg
```

### 4. Servo Stiffness Calibration
Measure actual servo response to tune `ServoModel`:
```python
# In torque_converter.py
KP_DEFAULT = 3.0    # Nm/rad - measure this!
```

## Implementation Notes

### Position-Controlled Servos
Mini Pupper uses position-controlled servos, not torque-controlled. We achieve "virtual torque control" by:
1. Modeling the servo as a spring: `τ = Kp * (θ_cmd - θ_actual)`
2. Computing position offset: `Δθ = τ_desired / Kp`
3. Adding offset to base position

### Encoder Simulation
Since Mini Pupper doesn't have joint encoders, we estimate joint positions from:
- Commanded positions (assuming servo tracks well)
- Could be improved with external encoders

### Coordinate Transform
The IMU gives roll/pitch in body frame. We transform to the diagonal axis:
```python
θ_body = roll * cos(ψ) + pitch * sin(ψ)
```
where `ψ` is the angle of the diagonal axis.

## Files

- `__init__.py` - Package exports
- `config.py` - All parameters and tuning values
- `state_estimator.py` - Sensor fusion and state estimation
- `lqr_controller.py` - LQR control with cooperative distribution
- `torque_converter.py` - Torque to position mapping
- `safety.py` - Safety monitoring and logging
- `controller.py` - Main control loop
- `README.md` - This file
