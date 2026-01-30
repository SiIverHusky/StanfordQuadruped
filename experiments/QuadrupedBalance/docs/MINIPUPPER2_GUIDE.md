# Implementing Diagonal Balance Control on Mini Pupper 2

A comprehensive guide for porting the QuadrupedBalance.jl control framework to the Mini Pupper 2 quadruped robot.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mini Pupper 2 vs Unitree A1: Key Differences](#mini-pupper-2-vs-unitree-a1-key-differences)
3. [Understanding the Control Problem](#understanding-the-control-problem)
4. [Lessons from the Previous LQR Attempt](#lessons-from-the-previous-lqr-attempt)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Step 1: Hardware Abstraction Layer](#step-1-hardware-abstraction-layer)
7. [Step 2: Kinematics and Dynamics Model](#step-2-kinematics-and-dynamics-model)
8. [Step 3: State Estimation](#step-3-state-estimation)
9. [Step 4: Equilibrium Finding](#step-4-equilibrium-finding)
10. [Step 5: Controller Design](#step-5-controller-design)
11. [Step 6: Simulation and Testing](#step-6-simulation-and-testing)
12. [Step 7: Hardware Deployment](#step-7-hardware-deployment)
13. [Recommended Python Libraries](#recommended-python-libraries)
14. [Code Examples](#code-examples)
15. [Troubleshooting Guide](#troubleshooting-guide)

---

## Introduction

This guide explains how to implement the maximal-coordinate LQR balance control from **QuadrupedBalance.jl** on the **Mini Pupper 2** robot. The goal is to achieve stable diagonal stance (balancing on two diagonal legs, e.g., Front-Right + Back-Left).

### Why This is Challenging

When a quadruped stands on two diagonal legs:
- There is **no support polygon** (only a line between feet)
- The robot becomes an **inverted pendulum**
- Traditional static stability (ZMP) doesn't apply
- Active feedback control is **required** to maintain balance

---

## Mini Pupper 2 vs Unitree A1: Key Differences

| Parameter | Mini Pupper 2 | Unitree A1 | Impact |
|-----------|---------------|------------|--------|
| **Mass** | ~0.88 kg | ~12 kg | Faster dynamics, less inertia |
| **Body Length** | 0.276 m | 0.366 m | Shorter pendulum |
| **Leg Length (L1+L2)** | 0.239 m | 0.4 m | Lower CoM |
| **LEG_FB** | 0.10 m | 0.183 m | Narrower diagonal base |
| **LEG_LR** | 0.04 m | 0.047 m | Similar side offset |
| **Actuators** | Hobby servos | Proprietary motors | Lower torque, no torque control |
| **Sensors** | ESP32 IMU | High-grade IMU | More noise, slower update |
| **Control Rate** | ~100-200 Hz | 500+ Hz | Slower response |

### Critical Hardware Differences

1. **No Direct Torque Control**: Mini Pupper servos are position-controlled, not torque-controlled. This fundamentally changes how we apply the LQR output.

2. **Limited IMU Quality**: The ESP32's integrated IMU is noisier than research-grade sensors. Kalman filtering is essential.

3. **Lower Servo Bandwidth**: Hobby servos can't track high-frequency commands. The effective control bandwidth is limited.

4. **No Joint Encoders**: We don't have direct feedback of actual joint angles, only commanded positions.

---

## Understanding the Control Problem

### The QuadrupedBalance.jl Approach

The original Julia implementation uses:

1. **Full-body rigid body dynamics** (37-dimensional state: quaternion + position + 12 joints + 18 velocities)
2. **Contact constraints** to pin feet to the ground
3. **Maximal coordinate LQR** with constraint enforcement
4. **Offline equilibrium finding** via IPOPT
5. **Online state feedback**: $\mathbf{u} = \mathbf{u}^* - \mathbf{K}(\mathbf{x} \ominus \mathbf{x}^*)$

### Simplification for Mini Pupper

For Mini Pupper 2, we can use a **simplified centroidal model** because:
- The legs are lightweight compared to the body
- We have position control, not torque control
- The dynamics are dominated by the body orientation

---

## Lessons from the Previous LQR Attempt

The `lqr_diagonal_balance` experiment revealed several key issues:

### What Worked

1. **6-DOF Decomposition**: Breaking the problem into:
   - Part A: Balance around diagonal axis (LQR)
   - Part B: Along-diagonal disturbance (PID)
   - Part C: Height maintenance (PID)

2. **Virtual Model Control (VMC)**: Mapping between diagonal-axis coordinates and actual roll/pitch joints.

3. **State Estimation**: Kalman filter for IMU noise reduction.

### What Didn't Work

1. **Torque-to-Position Gap**: The LQR computes torques, but Mini Pupper only accepts positions. The naive conversion:
   ```python
   position_cmd = equilibrium_position + K_pos * torque_cmd
   ```
   doesn't properly account for dynamics.

2. **Model Mismatch**: The simplified biped model didn't capture the actual Mini Pupper dynamics well enough.

3. **Servo Bandwidth**: The 200 Hz control loop was too fast for the servos to track, causing oscillation.

4. **IMU Latency**: By the time IMU data was processed, the robot had already moved significantly.

### Key Insight from the Attempt

From [controller.py](../lqr_diagonal_balance/controller.py#L260-L280), the most successful approach was **direct foot position correction**:

```python
# Get current tilt (already baseline-compensated)
roll = imu_data.get('roll', 0)   # Radians
pitch = imu_data.get('pitch', 0)  # Radians

# Calculate desired corrections
desired_y = -roll * self.balance_gain_y   # Opposite direction to tilt
desired_x = -pitch * self.balance_gain_x  # Opposite direction to tilt
```

This bypasses the torque computation entirely and maps IMU feedback directly to foot position adjustments.

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Create dynamics model for Mini Pupper
2. Implement forward/inverse kinematics
3. Set up simulation environment (PyBullet or MuJoCo)

### Phase 2: Equilibrium Finding (Week 3)
1. Port the IPOPT equilibrium finder to Python (using CasADi)
2. Find equilibrium poses for diagonal stance
3. Validate in simulation

### Phase 3: Controller Design (Weeks 4-5)
1. Linearize dynamics at equilibrium
2. Compute LQR gains
3. Design torque-to-position mapping
4. Implement Kalman filter for state estimation

### Phase 4: Simulation Testing (Week 6)
1. Test in PyBullet with realistic servo models
2. Tune gains with disturbance tests
3. Add sensor noise to validate robustness

### Phase 5: Hardware Deployment (Weeks 7-8)
1. Implement hardware interface
2. Careful bench testing
3. Gradual free-standing tests

---

## Step 1: Hardware Abstraction Layer

Create a unified interface that works for both simulation and hardware:

```python
# hardware_interface.py

from abc import ABC, abstractmethod
import numpy as np

class RobotInterface(ABC):
    """Abstract interface for Mini Pupper hardware."""
    
    @abstractmethod
    def set_joint_positions(self, positions: np.ndarray):
        """
        Set desired joint positions.
        
        Args:
            positions: (3, 4) array of joint angles [abduction, hip, knee] x [FR, FL, BR, BL]
        """
        pass
    
    @abstractmethod
    def get_imu_data(self) -> dict:
        """
        Get IMU measurements.
        
        Returns:
            Dict with 'roll', 'pitch', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z'
        """
        pass
    
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions (if available)."""
        pass


class MiniPupperHardware(RobotInterface):
    """Real Mini Pupper 2 hardware interface."""
    
    def __init__(self):
        from MangDang.mini_pupper.HardwareInterface import HardwareInterface
        from MangDang.mini_pupper.Config import Configuration
        from MangDang.mini_pupper.ESP32Interface import ESP32Interface
        
        self.hardware = HardwareInterface()
        self.config = Configuration()
        self.esp32 = ESP32Interface()
        self._last_commanded = np.zeros((3, 4))
    
    def set_joint_positions(self, positions: np.ndarray):
        self.hardware.set_actuator_postions(positions)
        self._last_commanded = positions.copy()
    
    def get_imu_data(self) -> dict:
        data = self.esp32.imu_get_data()
        
        # Convert accelerometer to angles
        ax, ay, az = data['ax'], data['ay'], data['az']
        roll = np.arctan2(ay, np.sqrt(ax**2 + az**2))
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        return {
            'roll': roll,
            'pitch': pitch,
            'yaw': 0.0,  # Requires magnetometer
            'gyro_x': data.get('gx', 0),
            'gyro_y': data.get('gy', 0),
            'gyro_z': data.get('gz', 0),
        }
    
    def get_joint_positions(self) -> np.ndarray:
        # No encoders - return last commanded
        return self._last_commanded


class SimulatedMiniPupper(RobotInterface):
    """PyBullet simulation of Mini Pupper."""
    
    def __init__(self, urdf_path: str = None):
        import pybullet as p
        import pybullet_data
        
        self.p = p
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load robot
        self.plane = p.loadURDF("plane.urdf")
        # You'll need a Mini Pupper URDF
        self.robot = p.loadURDF(urdf_path or "minipupper.urdf", [0, 0, 0.2])
        
        # Joint indices mapping
        self._setup_joints()
    
    def set_joint_positions(self, positions: np.ndarray):
        for i, joint_id in enumerate(self.joint_ids):
            self.p.setJointMotorControl2(
                self.robot, joint_id,
                self.p.POSITION_CONTROL,
                targetPosition=positions.flat[i],
                force=1.0  # Max torque for servo
            )
    
    def get_imu_data(self) -> dict:
        pos, orn = self.p.getBasePositionAndOrientation(self.robot)
        vel, ang_vel = self.p.getBaseVelocity(self.robot)
        
        # Convert quaternion to Euler
        euler = self.p.getEulerFromQuaternion(orn)
        
        return {
            'roll': euler[0],
            'pitch': euler[1],
            'yaw': euler[2],
            'gyro_x': ang_vel[0],
            'gyro_y': ang_vel[1],
            'gyro_z': ang_vel[2],
        }
```

---

## Step 2: Kinematics and Dynamics Model

### Mini Pupper Kinematics

The kinematic constants from [Config.py](../../pupper/Config.py):

```python
# mini_pupper_model.py

import numpy as np

class MiniPupperParams:
    """Physical parameters of Mini Pupper 2."""
    
    # Geometry (meters)
    LEG_FB = 0.10       # Front-back distance from center to leg
    LEG_LR = 0.04       # Left-right distance from center to leg plane
    LEG_L1 = 0.1235     # Upper leg length
    LEG_L2 = 0.115      # Lower leg length
    ABDUCTION_OFFSET = 0.03
    
    # Masses (kg)
    FRAME_MASS = 0.560
    MODULE_MASS = 0.080  # Per hip module
    LEG_MASS = 0.030     # Per leg
    TOTAL_MASS = FRAME_MASS + 4 * (MODULE_MASS + LEG_MASS)  # 0.88 kg
    
    # Diagonal stance geometry
    DIAGONAL_LENGTH = np.sqrt((2 * LEG_FB)**2 + (2 * LEG_LR)**2)  # ~0.216 m
    DIAGONAL_ANGLE = np.arctan2(2 * LEG_LR, 2 * LEG_FB)  # ~0.38 rad (~22°)
    
    # Default standing height
    DEFAULT_HEIGHT = 0.16  # meters
    
    # Leg origins in body frame
    LEG_ORIGINS = np.array([
        [LEG_FB, LEG_FB, -LEG_FB, -LEG_FB],   # X: +front, -back
        [-LEG_LR, LEG_LR, -LEG_LR, LEG_LR],   # Y: -right, +left
        [0, 0, 0, 0],                          # Z
    ])


def forward_kinematics(joint_angles: np.ndarray, params: MiniPupperParams = None) -> np.ndarray:
    """
    Compute foot positions in body frame from joint angles.
    
    Args:
        joint_angles: (3, 4) array [abduction, hip, knee] x [FR, FL, BR, BL]
        params: Robot parameters
        
    Returns:
        (3, 4) array of foot positions in body frame
    """
    if params is None:
        params = MiniPupperParams()
    
    foot_positions = np.zeros((3, 4))
    
    for leg in range(4):
        abd, hip, knee = joint_angles[:, leg]
        
        # Leg origin
        origin = params.LEG_ORIGINS[:, leg]
        
        # Abduction offset (y-direction)
        abd_offset = params.ABDUCTION_OFFSET * (1 if leg in [1, 3] else -1)
        
        # Forward kinematics in leg frame
        # x = forward, y = lateral, z = down
        x = -params.LEG_L1 * np.sin(hip) - params.LEG_L2 * np.sin(hip + knee)
        y_leg = params.LEG_L1 * np.cos(hip) * np.sin(abd) + params.LEG_L2 * np.cos(hip + knee) * np.sin(abd)
        z = -params.LEG_L1 * np.cos(hip) * np.cos(abd) - params.LEG_L2 * np.cos(hip + knee) * np.cos(abd)
        
        # Transform to body frame
        foot_positions[0, leg] = origin[0] + x
        foot_positions[1, leg] = origin[1] + abd_offset + y_leg
        foot_positions[2, leg] = z
    
    return foot_positions


def foot_jacobian(joint_angles: np.ndarray, leg_index: int, params: MiniPupperParams = None) -> np.ndarray:
    """
    Compute the foot velocity Jacobian for a single leg.
    
    Args:
        joint_angles: (3,) array [abduction, hip, knee]
        leg_index: 0=FR, 1=FL, 2=BR, 3=BL
        params: Robot parameters
        
    Returns:
        (3, 3) Jacobian matrix where v_foot = J @ joint_velocities
    """
    # Use numerical differentiation for simplicity
    eps = 1e-6
    J = np.zeros((3, 3))
    
    full_angles = np.zeros((3, 4))
    full_angles[:, leg_index] = joint_angles
    
    f0 = forward_kinematics(full_angles, params)[:, leg_index]
    
    for i in range(3):
        full_angles_pert = full_angles.copy()
        full_angles_pert[i, leg_index] += eps
        f1 = forward_kinematics(full_angles_pert, params)[:, leg_index]
        J[:, i] = (f1 - f0) / eps
    
    return J
```

### Centroidal Dynamics Model

For balance control, we model the robot as a single rigid body:

```python
# centroidal_model.py

import numpy as np
from scipy.spatial.transform import Rotation

class CentroidalModel:
    """
    Simplified centroidal dynamics model for Mini Pupper.
    
    State: x = [position(3), velocity(3), quaternion(4), angular_velocity(3)]  # 13-dim
    """
    
    def __init__(self, params: MiniPupperParams = None):
        self.params = params or MiniPupperParams()
        
        # Approximate body inertia (box with dimensions L x W x H)
        L, W, H = 0.20, 0.10, 0.05  # meters
        m = self.params.TOTAL_MASS
        
        self.inertia = np.diag([
            m/12 * (W**2 + H**2),  # Ixx
            m/12 * (L**2 + H**2),  # Iyy
            m/12 * (L**2 + W**2),  # Izz
        ])
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        self.gravity = np.array([0, 0, -9.81])
    
    def dynamics(self, state: np.ndarray, foot_positions: np.ndarray, 
                 foot_forces: np.ndarray) -> np.ndarray:
        """
        Compute state derivative.
        
        Args:
            state: (13,) array [pos(3), vel(3), quat(4), omega(3)]
            foot_positions: (3, n_contact) positions of feet in contact (world frame)
            foot_forces: (3, n_contact) forces at each foot (world frame)
            
        Returns:
            (13,) state derivative
        """
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        
        # Rotation matrix (body to world)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        
        # Linear dynamics
        total_force = foot_forces.sum(axis=1) + self.params.TOTAL_MASS * self.gravity
        acc = total_force / self.params.TOTAL_MASS
        
        # Angular dynamics
        total_torque = np.zeros(3)
        for i in range(foot_forces.shape[1]):
            r = foot_positions[:, i] - pos  # Moment arm
            total_torque += np.cross(r, foot_forces[:, i])
        
        # Transform to body frame
        torque_body = R.T @ total_torque
        omega_body = R.T @ omega
        
        # Euler's equation in body frame
        omega_dot_body = self.inertia_inv @ (torque_body - np.cross(omega_body, self.inertia @ omega_body))
        omega_dot = R @ omega_dot_body
        
        # Quaternion derivative
        quat_dot = 0.5 * self._quat_multiply(quat, np.array([0, omega[0], omega[1], omega[2]]))
        
        return np.concatenate([vel, acc, quat_dot, omega_dot])
    
    def _quat_multiply(self, q1, q2):
        """Quaternion multiplication (scalar-first convention)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def linearize(self, x_eq: np.ndarray, foot_positions: np.ndarray) -> tuple:
        """
        Linearize dynamics around equilibrium.
        
        Uses numerical differentiation for simplicity.
        
        Returns:
            (A, B): Continuous-time state-space matrices
        """
        import numdifftools as nd
        
        n_feet = foot_positions.shape[1]
        
        def f(xu):
            x = xu[:13]
            u = xu[13:].reshape(3, n_feet)
            return self.dynamics(x, foot_positions, u)
        
        xu_eq = np.concatenate([x_eq, np.zeros(3 * n_feet)])
        J = nd.Jacobian(f)(xu_eq)
        
        A = J[:, :13]
        B = J[:, 13:]
        
        return A, B
```

---

## Step 3: State Estimation

A Kalman filter is essential for handling IMU noise:

```python
# state_estimator.py

import numpy as np

class BalanceStateEstimator:
    """
    Kalman filter for balance state estimation.
    
    Estimates: [roll, pitch, roll_rate, pitch_rate]
    """
    
    def __init__(self, dt: float = 0.005):
        self.dt = dt
        self.n_states = 4
        self.n_meas = 4  # roll, pitch from accel; roll_rate, pitch_rate from gyro
        
        # State: [roll, pitch, roll_rate, pitch_rate]
        self.x = np.zeros(self.n_states)
        
        # State covariance
        self.P = np.eye(self.n_states) * 0.1
        
        # Process model: constant rate + noise
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        
        # Process noise (tune these!)
        self.Q = np.diag([0.001, 0.001, 0.01, 0.01])
        
        # Measurement model: observe all states
        self.H = np.eye(self.n_states)
        
        # Measurement noise
        # Accel-derived angles: ~0.02 rad std
        # Gyro: ~0.01 rad/s std
        self.R = np.diag([0.02**2, 0.02**2, 0.01**2, 0.01**2])
    
    def predict(self):
        """Prediction step."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, imu_data: dict):
        """
        Update step with IMU measurement.
        
        Args:
            imu_data: Dict with 'roll', 'pitch', 'gyro_x', 'gyro_y'
        """
        z = np.array([
            imu_data['roll'],
            imu_data['pitch'],
            imu_data['gyro_x'],  # Assuming body-frame gyro aligned with roll
            imu_data['gyro_y'],  # Assuming body-frame gyro aligned with pitch
        ])
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(self.n_states) - K @ self.H) @ self.P
    
    def get_state(self) -> dict:
        """Get current state estimate."""
        return {
            'roll': self.x[0],
            'pitch': self.x[1],
            'roll_rate': self.x[2],
            'pitch_rate': self.x[3],
        }
```

---

## Step 4: Equilibrium Finding

Find a static equilibrium for diagonal stance using optimization:

```python
# equilibrium_finder.py

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List

def find_diagonal_equilibrium(
    support_legs: List[int],  # e.g., [0, 3] for FR+BL
    target_height: float = 0.14,
    params: MiniPupperParams = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find equilibrium pose for diagonal stance.
    
    Args:
        support_legs: Indices of legs in contact (0=FR, 1=FL, 2=BR, 3=BL)
        target_height: Desired body height above ground
        params: Robot parameters
        
    Returns:
        joint_angles: (3, 4) equilibrium joint angles
        foot_forces: (3, 2) contact forces at support feet
    """
    params = params or MiniPupperParams()
    
    # Decision variables: [joint_angles(12), foot_forces(6)]
    # joint_angles for all 4 legs, forces only for support legs
    
    def objective(x):
        """Minimize deviation from default pose and control effort."""
        angles = x[:12].reshape(3, 4)
        forces = x[12:].reshape(3, 2)
        
        # Default angles (slightly bent knees)
        default = np.zeros((3, 4))
        default[2, :] = -0.5  # Knee angle
        
        # Cost: angle deviation + force magnitude
        cost = 0.01 * np.sum((angles - default)**2)
        cost += 0.001 * np.sum(forces**2)
        
        return cost
    
    def equilibrium_constraint(x):
        """Static equilibrium: sum of forces = -mg, sum of torques = 0."""
        angles = x[:12].reshape(3, 4)
        forces = x[12:].reshape(3, 2)
        
        # Compute foot positions
        foot_pos = forward_kinematics(angles, params)
        
        # Body CoM at origin, height z = target_height
        # Foot positions are relative to body, so world z = body_z + foot_z
        # At equilibrium, body is at height target_height
        
        constraints = []
        
        # Force balance: sum(F) = [0, 0, mg]
        total_force = forces.sum(axis=1)
        constraints.extend([
            total_force[0],  # Fx = 0
            total_force[1],  # Fy = 0
            total_force[2] - params.TOTAL_MASS * 9.81,  # Fz = mg
        ])
        
        # Torque balance: sum(r x F) = 0
        total_torque = np.zeros(3)
        for i, leg in enumerate(support_legs):
            r = foot_pos[:, leg]  # Moment arm from CoM
            total_torque += np.cross(r, forces[:, i])
        constraints.extend(total_torque.tolist())
        
        return np.array(constraints)
    
    def foot_height_constraint(x):
        """Support feet must be at ground level."""
        angles = x[:12].reshape(3, 4)
        foot_pos = forward_kinematics(angles, params)
        
        # Support feet at z = -target_height (ground level from body frame)
        constraints = []
        for leg in support_legs:
            constraints.append(foot_pos[2, leg] + target_height)
        
        return np.array(constraints)
    
    def contact_force_constraint(x):
        """Contact forces must be compressive (Fz > 0) and within friction cone."""
        forces = x[12:].reshape(3, 2)
        mu = 0.4  # Friction coefficient
        
        constraints = []
        for i in range(2):
            # Fz > 0 (compressive)
            constraints.append(forces[2, i])
            # |Fx| < mu * Fz, |Fy| < mu * Fz (friction cone)
            constraints.append(mu * forces[2, i] - np.abs(forces[0, i]))
            constraints.append(mu * forces[2, i] - np.abs(forces[1, i]))
        
        return np.array(constraints)
    
    # Initial guess
    x0 = np.zeros(18)
    x0[12 + 2] = params.TOTAL_MASS * 9.81 / 2  # Half weight on each foot
    x0[12 + 5] = params.TOTAL_MASS * 9.81 / 2
    
    # Solve
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=[
            {'type': 'eq', 'fun': equilibrium_constraint},
            {'type': 'eq', 'fun': foot_height_constraint},
            {'type': 'ineq', 'fun': contact_force_constraint},
        ],
        options={'maxiter': 1000, 'disp': True}
    )
    
    if not result.success:
        print(f"Warning: Equilibrium finding did not converge: {result.message}")
    
    joint_angles = result.x[:12].reshape(3, 4)
    foot_forces = result.x[12:].reshape(3, 2)
    
    return joint_angles, foot_forces
```

---

## Step 5: Controller Design

### Option A: Full LQR (Recommended for Research)

```python
# lqr_controller.py

import numpy as np
from scipy.linalg import solve_continuous_are, expm

class MaximalLQRController:
    """
    Maximal coordinate LQR with contact constraints.
    
    Based on QuadrupedBalance.jl methodology.
    """
    
    def __init__(self, model: CentroidalModel, equilibrium: dict, support_legs: list):
        self.model = model
        self.x_eq = equilibrium['state']
        self.u_eq = equilibrium['forces']
        self.support_legs = support_legs
        
        # Get foot positions at equilibrium
        self.foot_positions_eq = equilibrium['foot_positions']
        
        # Linearize
        self.A, self.B = model.linearize(self.x_eq, self.foot_positions_eq)
        
        # LQR weights
        self.Q = np.diag([
            10.0, 10.0, 100.0,   # Position (z more important)
            1.0, 1.0, 1.0,       # Velocity
            500.0, 500.0, 50.0, 10.0,  # Quaternion (roll/pitch critical)
            10.0, 10.0, 1.0,     # Angular velocity
        ])
        
        self.R = np.eye(6) * 0.01  # Low control cost
        
        # Solve CARE
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P
        
        # Discretize for implementation
        self.dt = 0.005
        self._discretize()
    
    def _discretize(self):
        """Discretize continuous-time system."""
        n = self.A.shape[0]
        m = self.B.shape[1]
        
        # Matrix exponential method
        M = np.zeros((n + m, n + m))
        M[:n, :n] = self.A * self.dt
        M[:n, n:] = self.B * self.dt
        
        expM = expm(M)
        self.Ad = expM[:n, :n]
        self.Bd = expM[:n, n:]
    
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control input.
        
        Args:
            state: Current state (13,)
            
        Returns:
            foot_forces: (3, 2) forces to apply
        """
        # State error (handle quaternion specially)
        error = self._compute_error(state, self.x_eq)
        
        # LQR control law
        delta_u = -self.K @ error
        
        # Add equilibrium forces
        u = self.u_eq.flatten() + delta_u
        
        return u.reshape(3, -1)
    
    def _compute_error(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Compute state error with proper quaternion handling."""
        error = x - x_ref
        
        # Quaternion error (use logarithm map)
        q = x[6:10]
        q_ref = x_ref[6:10]
        q_err = self._quat_multiply(q, self._quat_conjugate(q_ref))
        
        # Convert to axis-angle (small angle approximation)
        if q_err[0] < 0:
            q_err = -q_err
        error[6:10] = 2 * q_err[1:4]  # Use vector part as error
        
        return error[:13]  # Return without last component
```

### Option B: Simplified Position-Based Control (Recommended for Initial Testing)

This approach, derived from the lessons learned in `lqr_diagonal_balance`, maps IMU feedback directly to foot position corrections:

```python
# simple_balance_controller.py

import numpy as np
from typing import Tuple

class SimpleBalanceController:
    """
    Simplified balance controller using foot position adjustment.
    
    This bypasses torque computation and directly maps IMU readings
    to foot position corrections, which is more suitable for
    position-controlled servos.
    """
    
    def __init__(self, params: MiniPupperParams = None):
        self.params = params or MiniPupperParams()
        
        # Gains (meters of foot shift per radian of tilt)
        self.kp_roll = 0.08   # Lateral correction gain
        self.kp_pitch = 0.08  # Forward/back correction gain
        self.kd_roll = 0.02   # Derivative gain
        self.kd_pitch = 0.02
        
        # Rate limiting (max correction rate in m/s)
        self.max_rate = 0.1
        
        # Previous corrections for rate limiting
        self.last_correction = np.zeros(2)  # [x, y]
        self.last_time = None
    
    def compute_correction(self, state: dict, dt: float) -> Tuple[float, float]:
        """
        Compute foot position correction.
        
        Args:
            state: Dict with 'roll', 'pitch', 'roll_rate', 'pitch_rate'
            dt: Time step
            
        Returns:
            (delta_x, delta_y): Foot position corrections in meters
        """
        roll = state['roll']
        pitch = state['pitch']
        roll_rate = state.get('roll_rate', 0)
        pitch_rate = state.get('pitch_rate', 0)
        
        # PD control: correction opposes tilt
        delta_y = -(self.kp_roll * roll + self.kd_roll * roll_rate)
        delta_x = -(self.kp_pitch * pitch + self.kd_pitch * pitch_rate)
        
        # Rate limiting
        correction = np.array([delta_x, delta_y])
        delta = correction - self.last_correction
        max_delta = self.max_rate * dt
        
        if np.linalg.norm(delta) > max_delta:
            delta = delta / np.linalg.norm(delta) * max_delta
            correction = self.last_correction + delta
        
        self.last_correction = correction
        
        return correction[0], correction[1]
    
    def apply_to_stance(self, base_stance: np.ndarray, 
                        support_legs: list,
                        delta_x: float, 
                        delta_y: float) -> np.ndarray:
        """
        Apply corrections to stance matrix.
        
        Args:
            base_stance: (3, 4) base foot positions
            support_legs: Indices of legs in contact
            delta_x: Forward/back correction
            delta_y: Lateral correction
            
        Returns:
            Modified stance matrix
        """
        stance = base_stance.copy()
        
        for leg in support_legs:
            stance[0, leg] += delta_x
            stance[1, leg] += delta_y
        
        return stance
```

---

## Step 6: Simulation and Testing

### PyBullet Simulation

```python
# simulation.py

import pybullet as p
import numpy as np
import time

def run_balance_simulation(
    controller,
    duration: float = 10.0,
    dt: float = 0.005,
    disturbance: dict = None
):
    """
    Run balance simulation in PyBullet.
    
    Args:
        controller: Balance controller instance
        duration: Simulation duration in seconds
        dt: Time step
        disturbance: Optional disturbance parameters
    """
    # Initialize simulation
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    
    # Load robot (you'll need a URDF)
    # For now, approximate with a simple box + legs
    
    # Simulation loop
    t = 0
    history = []
    
    while t < duration:
        # Get state
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        vel, ang_vel = p.getBaseVelocity(robot_id)
        
        euler = p.getEulerFromQuaternion(orn)
        
        state = {
            'roll': euler[0],
            'pitch': euler[1],
            'roll_rate': ang_vel[0],
            'pitch_rate': ang_vel[1],
        }
        
        # Compute control
        delta_x, delta_y = controller.compute_correction(state, dt)
        
        # Apply to legs (convert to joint angles via IK)
        # ...
        
        # Apply disturbance if specified
        if disturbance and t > 1.0:
            # Apply impulse
            pass
        
        # Step simulation
        p.stepSimulation()
        t += dt
        
        # Log
        history.append({
            't': t,
            'roll': euler[0],
            'pitch': euler[1],
            'delta_x': delta_x,
            'delta_y': delta_y,
        })
        
        time.sleep(dt)  # Real-time visualization
    
    p.disconnect()
    return history
```

---

## Step 7: Hardware Deployment

### Safety Checklist

Before running on hardware:

- [ ] Test all components individually (IMU, servos)
- [ ] Verify servo calibration
- [ ] Set conservative joint limits
- [ ] Implement emergency stop (Ctrl+C handler)
- [ ] Start with robot secured/supported
- [ ] Have someone ready to catch the robot
- [ ] Clear the test area

### Main Loop

```python
# main.py

import numpy as np
import time
import signal
import sys

class DiagonalBalanceRunner:
    """Main runner for diagonal balance experiment."""
    
    def __init__(self, mode: str = 'simulation'):
        self.mode = mode
        self.running = False
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._stop)
        
        # Initialize components
        if mode == 'simulation':
            self.robot = SimulatedMiniPupper()
        else:
            self.robot = MiniPupperHardware()
        
        self.estimator = BalanceStateEstimator()
        self.controller = SimpleBalanceController()
        
        # Configuration
        self.support_legs = [1, 2]  # FL + BR (or [0, 3] for FR + BL)
        self.lift_legs = [0, 3]
        
        self.dt = 0.005  # 200 Hz
        
        # Find equilibrium
        self.eq_angles, self.eq_forces = find_diagonal_equilibrium(
            self.support_legs, target_height=0.14
        )
    
    def _stop(self, signum, frame):
        """Handle Ctrl+C."""
        print("\nStopping...")
        self.running = False
    
    def run(self, duration: float = 10.0):
        """Run the balance experiment."""
        print(f"Starting diagonal balance ({self.mode} mode)")
        print(f"Support legs: {self.support_legs}")
        print(f"Duration: {duration}s")
        print("Press Ctrl+C to stop\n")
        
        # Start in normal stance
        self._go_to_normal_stance()
        time.sleep(1.0)
        
        # Gradually lift legs
        self._lift_legs(duration=1.0)
        
        # Balance loop
        self.running = True
        t_start = time.time()
        
        while self.running and (time.time() - t_start) < duration:
            t_loop = time.time()
            
            # Read sensors
            imu_data = self.robot.get_imu_data()
            
            # Update estimator
            self.estimator.predict()
            self.estimator.update(imu_data)
            state = self.estimator.get_state()
            
            # Safety check
            if abs(state['roll']) > np.deg2rad(45) or abs(state['pitch']) > np.deg2rad(45):
                print("SAFETY: Excessive tilt detected!")
                break
            
            # Compute control
            delta_x, delta_y = self.controller.compute_correction(state, self.dt)
            
            # Apply to stance
            base_stance = self._get_diagonal_stance()
            corrected_stance = self.controller.apply_to_stance(
                base_stance, self.support_legs, delta_x, delta_y
            )
            
            # Convert to joint angles
            from pupper.Kinematics import four_legs_inverse_kinematics
            from pupper.Config import Configuration
            config = Configuration()
            joint_angles = four_legs_inverse_kinematics(corrected_stance, config)
            
            # Send to robot
            self.robot.set_joint_positions(joint_angles)
            
            # Timing
            elapsed = time.time() - t_loop
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
        
        # Return to normal stance
        print("\nReturning to normal stance...")
        self._lower_legs(duration=1.0)
        self._go_to_normal_stance()
        
        print("Done!")
    
    def _get_diagonal_stance(self) -> np.ndarray:
        """Get base diagonal stance with lifted legs."""
        from pupper.Config import Configuration
        config = Configuration()
        stance = config.default_stance.copy()
        
        # Lower support legs, raise lift legs
        for leg in self.support_legs:
            stance[2, leg] = -0.14  # Lower
        for leg in self.lift_legs:
            stance[2, leg] = -0.10  # Higher
        
        return stance


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['simulation', 'hardware'], default='simulation')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()
    
    runner = DiagonalBalanceRunner(mode=args.mode)
    runner.run(duration=args.duration)
```

---

## Recommended Python Libraries

| Purpose | Library | Notes |
|---------|---------|-------|
| Dynamics | `pinocchio` | Best for rigid body dynamics |
| Simulation | `pybullet` | Easy to use, good visualization |
| Optimization | `casadi` | For equilibrium finding (IPOPT backend) |
| Linear algebra | `numpy`, `scipy` | Standard tools |
| Autodiff | `jax` | For analytical Jacobians |
| Filtering | `filterpy` | Kalman filter implementations |
| Quaternions | `scipy.spatial.transform` | Rotation handling |

Install with:
```bash
pip install numpy scipy pybullet casadi filterpy
pip install pin  # pinocchio
```

---

## Troubleshooting Guide

### Problem: Robot oscillates rapidly
**Cause**: Control gains too high or servo bandwidth exceeded
**Solution**: 
- Reduce `kp_roll` and `kp_pitch` by 50%
- Add low-pass filter on corrections
- Reduce control frequency

### Problem: Robot slowly drifts and falls
**Cause**: Integral windup or sensor drift
**Solution**:
- Add integral term with anti-windup
- Calibrate IMU baseline at startup
- Check for sensor bias

### Problem: Jerky movements
**Cause**: Large discrete jumps in commands
**Solution**:
- Increase rate limiting (`max_rate`)
- Use trajectory interpolation
- Smooth IMU readings more aggressively

### Problem: Cannot find equilibrium
**Cause**: Infeasible target or poor initial guess
**Solution**:
- Try lower target height
- Check friction cone constraints
- Visualize foot positions

### Problem: Servo overheating
**Cause**: Continuous high-torque demands
**Solution**:
- Reduce balance gain magnitudes
- Add rest periods
- Check mechanical alignment

---

## References

1. **QuadrupedBalance.jl Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md)
2. **QuadrupedBalance.jl Code Reference**: [CODE_REFERENCE.md](CODE_REFERENCE.md)
3. **Porting Guide (General)**: [PORTING_GUIDE.md](PORTING_GUIDE.md)
4. **Paper**: "A New Balance Control Approach for Quadruped Robot with Diagonal Leg Support"
5. **Mini Pupper Hardware**: [pupper/Config.py](../../pupper/Config.py)
6. **Previous LQR Attempt**: [lqr_diagonal_balance/](../lqr_diagonal_balance/)

---

## Summary

Implementing diagonal balance on Mini Pupper 2 requires adapting the sophisticated QuadrupedBalance.jl approach to the limitations of hobby-grade hardware:

1. **Use position control instead of torque control** - Map IMU feedback to foot position corrections
2. **Account for servo dynamics** - Rate limit commands and don't expect high-frequency tracking
3. **Filter aggressively** - The IMU is noisy; Kalman filtering is essential
4. **Start simple** - Begin with the position-based controller before attempting full LQR
5. **Test extensively in simulation** - PyBullet is your friend
6. **Be patient** - Tuning will take time; document what works and what doesn't

Good luck! 🐕
