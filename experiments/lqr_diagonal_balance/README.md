# LQR-Based Diagonal Balance Experiment

This experiment implements the balance control approach from the paper:

> **"A New Balance Control Approach for Quadruped Robot with Diagonal Leg Support"**

## Key Concepts from the Paper

### 1. The Problem
When a quadruped robot stands on two diagonal legs, traditional static stability criteria (like ZMP) don't apply because there's no support polygon. The robot becomes an inverted pendulum that must actively balance.

### 2. The Solution: Decomposed 6-DOF Control

The paper divides the 6-DOF diagonal balance problem into three independent parts:

- **Part A**: Posture control around the diagonal axis (the main balance problem)
  - Uses **LQR control** (Linear Quadratic Regulator)
  - Based on Lagrangian dynamics of a biped model
  
- **Part B**: Control along the diagonal axis
  - Uses **PID control**
  - Simpler problem, no dynamic balancing needed
  
- **Part C**: Height control
  - Uses **PID control**
  - Maintains desired body height

### 3. Virtual Model Control (VMC)

The paper uses VMC to bridge between:
- A "virtual model" with joints aligned with the diagonal axis
- The actual robot with standard roll/pitch joints

This allows the controller to be designed for the simpler virtual model, then transformed to actual joint commands.

## Implementation Structure

```
lqr_diagonal_balance/
├── __init__.py              # Package exports
├── config.py                # All tunable parameters
├── dynamics.py              # Lagrangian dynamics model
├── lqr_controller.py        # LQR controller and Kalman filter
├── vmc.py                   # Virtual Model Control
├── decomposition.py         # Part A, B, C controllers
├── controller.py            # Main integration
├── run.py                   # Entry point script
└── README.md                # This file
```

## Usage

### Pure Simulation (No Hardware)
```bash
python -m experiments.lqr_diagonal_balance.run --mode simulation
```

### Bench Test (Robot Secured)
```bash
python -m experiments.lqr_diagonal_balance.run --mode bench
```

### Physical Operation
```bash
python -m experiments.lqr_diagonal_balance.run --mode physical
```

### Analyze LQR Design
```bash
python -m experiments.lqr_diagonal_balance.run --analyze
```

### Run Disturbance Tests
```bash
python -m experiments.lqr_diagonal_balance.run --disturbance-tests
```

## Key Differences from Original diagonal_balance Experiment

| Aspect | Original (PID) | New (LQR) |
|--------|---------------|-----------|
| Control method | PID for balance | LQR for Part A, PID for Parts B&C |
| Model | Ad-hoc tuning | Physics-based Lagrangian dynamics |
| State estimation | Simple filtering | Kalman filter |
| Joint mapping | Direct | Virtual Model Control (VMC) |
| Stability guarantee | None | LQR provides optimal gains |

## Tuning Guide

### LQR Weights (in config.py)

The Q matrix penalizes state deviations:
```python
Q = diag([θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot])
```
- Increase `θ_body` weight for tighter body angle control
- Increase velocity weights for more damping

The R matrix penalizes control effort:
```python
R = diag([τ_F, τ_H])
```
- Increase for gentler, slower response
- Decrease for more aggressive control (may cause oscillation)

### Safety Limits

```python
SafetyLimits.MAX_BODY_TILT = 35°  # Emergency stop threshold
SafetyLimits.MAX_ANGULAR_VELOCITY = 4.0  # rad/s
```

## Expected Results (from Paper)

The paper's simulation showed:
- Stable balance with pitch disturbances up to 3 rad/s
- Recovery from yaw disturbances up to 4 rad/s
- Recovery from lateral velocity disturbances up to 0.2 m/s
- Robustness to sensor noise (joint: 0.001 rad, IMU: 0.005 rad std)

## References

- Paper Figure 6: Control structure with state feedback
- Paper Figure 7: 6-DOF decomposition into Parts A, B, C
- Paper Equations (8-9): State space and LQR formulation
- Paper Equations (15-23): VMC transformation
