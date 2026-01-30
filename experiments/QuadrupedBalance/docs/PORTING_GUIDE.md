# Porting Guide: QuadrupedBalance to Other Robots and Languages

This guide explains how to port the QuadrupedBalance control framework to a different quadruped robot and/or programming language.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [What's Robot-Specific vs Generic](#whats-robot-specific-vs-generic)
3. [Porting to a New Robot (Same Language)](#porting-to-a-new-robot-same-language)
4. [Porting to Python](#porting-to-python)
5. [Porting to C++](#porting-to-c)
6. [Porting to MATLAB](#porting-to-matlab)
7. [Core Algorithm Pseudocode](#core-algorithm-pseudocode)
8. [Validation Checklist](#validation-checklist)

---

## Architecture Overview

The control framework has three main stages that are **language and robot agnostic**:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  1. EQUILIBRIUM     │     │  2. LQR GAIN        │     │  3. RUNTIME         │
│     FINDER          │────▶│     COMPUTATION     │────▶│     CONTROL         │
│                     │     │                     │     │                     │
│  • NLP optimization │     │  • Linearization    │     │  • State estimation │
│  • Static balance   │     │  • Discretization   │     │  • Error feedback   │
│  • Contact forces   │     │  • Riccati solve    │     │  • Torque command   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
        ▼                           ▼                           ▼
   ipopt_eq_point.toml       maximal_lqr_gain.txt         Real-time loop
   (offline)                 (offline)                    (online)
```

**Key insight**: Steps 1 and 2 run **offline** — they can be slow. Step 3 runs **online** and must be fast.

---

## What's Robot-Specific vs Generic

### Robot-Specific (Must Change)

| Component | What to Change | Where It's Used |
|-----------|----------------|-----------------|
| **URDF/Model** | Robot description file | Dynamics computation |
| **Kinematic parameters** | Link lengths, offsets | Forward kinematics |
| **Joint limits** | Position bounds | Equilibrium finder |
| **Mass/Inertia** | From URDF or CAD | Mass matrix |
| **Motor IDs** | Joint ordering convention | Control interface |
| **Leg naming** | FR/FL/RR/RL mapping | Foot indices |

### Generic (Can Reuse)

| Component | Description |
|-----------|-------------|
| **Quaternion math** | L(q), R(q), hat(), attitude error |
| **Constrained LQR** | Riccati iteration with constraints |
| **Semi-implicit Euler** | Integration with constraints |
| **Matrix exponential discretization** | Continuous → discrete dynamics |
| **NLP structure** | Equilibrium optimization formulation |

---

## Porting to a New Robot (Same Language)

### Step 1: Create Robot Model

Replace the URDF and create a new model struct:

```julia
# Original (A1)
struct UnitreeA1FullBody <: AbstractQuadruped
    rigidbody::RigidBodyModel 
end

# New robot (example: Spot)
struct BostonDynamicsSpot <: AbstractQuadruped
    rigidbody::RigidBodyModel 
end

function BostonDynamicsSpot(mech::Mechanism) 
    # Adjust control_indices based on your robot's DOF
    # A1 has 6 unactuated (floating base) + 12 actuated joints
    control_indices = Vector{Bool}([zeros(6); ones(12)])  # Modify if different
    model = RigidBodyModel(mech, control_indices)
    new(model)
end
```

### Step 2: Update Forward Kinematics

The analytical FK in `forward_kinematics.jl` is A1-specific. You have two options:

**Option A: Derive analytical FK for your robot**

```julia
# Your robot's kinematic parameters
const x_hip = 0.XXX    # Hip X offset from trunk
const y_hip = 0.XXX    # Hip Y offset from trunk  
const Δy_thigh = 0.XXX # Thigh offset from hip
const l_thigh = 0.XXX  # Thigh link length
const l_calf = 0.XXX   # Calf link length

function fk(q::AbstractVector)
    # Derive equations from your robot's kinematic chain
    # Use symbolic math or hand derivation
end
```

**Option B: Use numerical FK from URDF (easier, slightly slower)**

```julia
using RigidBodyDynamics

function fk_numerical(mech::Mechanism, state::MechanismState, foot_bodies::Vector{String})
    positions = Float64[]
    for foot_name in foot_bodies
        body = findbody(mech, foot_name)
        transform = relative_transform(state, root_frame(mech), default_frame(body))
        append!(positions, translation(transform))
    end
    return positions
end
```

### Step 3: Update Motor/Joint Mapping

Define your robot's joint ordering:

```julia
# A1 ordering
struct MotorIDs 
    FR_Hip::Int; FR_Thigh::Int; FR_Calf::Int
    FL_Hip::Int; FL_Thigh::Int; FL_Calf::Int 
    RR_Hip::Int; RR_Thigh::Int; RR_Calf::Int
    RL_Hip::Int; RL_Thigh::Int; RL_Calf::Int
end

# Your robot's ordering (example)
const MyRobotMotorIDs = MotorIDs(
    1, 2, 3,    # FR
    4, 5, 6,    # FL
    7, 8, 9,    # RR
    10, 11, 12  # RL
)
```

### Step 4: Adjust Equilibrium Finder

Update joint limits and initial guess:

```julia
# Load your URDF
mech = parse_urdf("path/to/your_robot.urdf", floating=true)

# Set initial pose guess (robot-specific)
x_guess = zeros(37)  # Adjust size if your robot has different DOF
x_guess[1] = 1.0     # Quaternion w component
x_guess[7] = 0.3     # Initial height (adjust for your robot)

# Joint initial guesses (adjust angles for your robot's home pose)
x_guess[[1,5,9] .+ 7] = [0.0, 0.8, -1.6]   # FR: hip, thigh, calf
# ... etc for other legs
```

### Step 5: Tune LQR Weights

The Q and R matrices need tuning for your robot:

```julia
# Position/attitude weights (usually similar)
Q_gains[1:3] = 1 ./ deg2rad(10.0)^2    # Attitude
Q_gains[4:6] = 1 ./ [0.01, 0.01, 0.5].^2  # Position

# Joint weights (robot-specific based on joint importance)
Q_gains[7:18] = 1 ./ deg2rad(15.0)^2   # Adjust per joint

# Control effort (based on motor torque limits)
R_gains = 1 ./ [τ_max_hip, τ_max_thigh, τ_max_calf, ...].^2
```

---

## Porting to Python

### Equivalent Libraries

| Julia | Python Equivalent |
|-------|-------------------|
| `RigidBodyDynamics.jl` | `pinocchio`, `pybullet`, `mujoco` |
| `ForwardDiff.jl` | `jax.grad`, `autograd`, `casadi` |
| `Rotations.jl` | `scipy.spatial.transform`, `pytransform3d` |
| `Ipopt.jl` | `cyipopt`, `casadi` |
| `StaticArrays.jl` | `numpy` (with care) |
| `MeshCat.jl` | `meshcat-python` |

### Core Implementation (Python + Pinocchio)

```python
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from scipy.linalg import expm, solve

class QuadrupedBalance:
    def __init__(self, urdf_path):
        # Load robot model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # State dimensions
        self.nq = self.model.nq  # Configuration dim
        self.nv = self.model.nv  # Velocity dim
        self.nu = 12             # Control dim (12 joints)
        
    def dynamics(self, x, u):
        """Compute xdot = f(x, u)"""
        q = x[:self.nq]
        v = x[self.nq:]
        
        # Mass matrix
        M = pin.crba(self.model, self.data, q)
        
        # Bias forces (Coriolis + gravity)
        h = pin.nle(self.model, self.data, q, v)
        
        # Control selection (only joints, not floating base)
        tau = np.zeros(self.nv)
        tau[6:] = u  # Assuming 6 DOF floating base
        
        # Forward dynamics
        v_dot = np.linalg.solve(M, tau - h)
        q_dot = pin.integrate(self.model, q, v) - q  # For quaternion
        
        return np.concatenate([q_dot, v_dot])
    
    def get_foot_positions(self, q, foot_frame_ids):
        """Forward kinematics for feet"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        positions = []
        for fid in foot_frame_ids:
            pos = self.data.oMf[fid].translation
            positions.extend(pos)
        return np.array(positions)
    
    def get_foot_jacobian(self, q, foot_frame_ids):
        """Contact Jacobian"""
        pin.computeJointJacobians(self.model, self.data, q)
        
        J = []
        for fid in foot_frame_ids:
            Ji = pin.getFrameJacobian(
                self.model, self.data, fid, 
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, :]  # Position only (3 rows)
            J.append(Ji)
        return np.vstack(J)


def quaternion_error(q_current, q_desired):
    """Compute attitude error (3-vector) using Cayley map"""
    r_curr = Rotation.from_quat(q_current[[1,2,3,0]])  # scipy uses xyzw
    r_des = Rotation.from_quat(q_desired[[1,2,3,0]])
    r_err = r_des.inv() * r_curr
    return r_err.as_rotvec()  # Axis-angle representation


def hat(v):
    """Skew-symmetric matrix from 3-vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def L_quat(q):
    """Left quaternion multiplication matrix"""
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w]
    ])


def discretize_dynamics(A, B, C, dt):
    """Matrix exponential discretization"""
    n, m, l = A.shape[0], B.shape[1], C.shape[1]
    
    # Build augmented matrix
    M = np.zeros((n + m + l, n + m + l))
    M[:n, :n] = A * dt
    M[:n, n:n+m] = B * dt
    M[:n, n+m:] = C * dt
    
    # Matrix exponential
    M_exp = expm(M)
    
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:n+m]
    Cd = M_exp[:n, n+m:]
    
    return Ad, Bd, Cd


def maximal_coordinate_lqr(Ad, Bd, Cd, D, Q, R, max_iter=10000, beta=1e-5):
    """
    Constrained LQR via backward Riccati iteration
    
    Args:
        Ad: Discrete state matrix (n x n)
        Bd: Discrete control matrix (n x m)  
        Cd: Discrete constraint force matrix (n x l)
        D: Constraint Jacobian (nc x n)
        Q: State cost (n x n)
        R: Control cost (m x m)
        
    Returns:
        K: Feedback gain matrix (m x n)
    """
    n = Q.shape[0]
    m = R.shape[0]
    l = Cd.shape[1]
    nc = D.shape[0]
    
    P = np.copy(Q)
    K_prev = np.zeros((m, n))
    
    for k in range(max_iter, 0, -1):
        # Build KKT system
        H = np.block([
            [R + Bd.T @ P @ Bd,           Bd.T @ P @ Cd,              Bd.T @ D.T],
            [Cd.T @ P @ Bd,     beta*np.eye(l) + Cd.T @ P @ Cd,       Cd.T @ D.T],
            [D @ Bd,                       D @ Cd,                -beta*np.eye(nc)]
        ])
        
        b = np.vstack([
            Bd.T @ P @ Ad,
            Cd.T @ P @ Ad,
            D @ Ad
        ])
        
        # Solve for gains
        KLM = solve(H, b)
        K = KLM[:m, :]
        L_ = KLM[m:m+l, :]
        M = KLM[m+l:, :]
        
        # Update P
        A_cl = Ad - Bd @ K - Cd @ L_
        P_new = Q + K.T @ R @ K + beta * L_.T @ L_ + A_cl.T @ P @ A_cl - M.T @ D @ A_cl
        
        # Check convergence
        if np.linalg.norm(K - K_prev) < 1e-8:
            print(f"Converged in {max_iter - k + 1} iterations")
            return K
        
        P = P_new
        K_prev = K.copy()
    
    print("Warning: Did not converge")
    return K


def semi_implicit_euler(robot, x, u, foot_positions_target, foot_ids, dt):
    """
    Constrained integration step
    
    Args:
        robot: QuadrupedBalance instance
        x: Current state
        u: Control input
        foot_positions_target: Desired foot positions (constraint)
        foot_ids: Frame IDs of feet in contact
        dt: Timestep
        
    Returns:
        x_next: Next state
        lambda_: Constraint forces
    """
    nq, nv = robot.nq, robot.nv
    q = x[:nq]
    v = x[nq:]
    
    # Dynamics terms
    M = pin.crba(robot.model, robot.data, q)
    h = pin.nle(robot.model, robot.data, q, v)
    
    # Contact Jacobian
    J = robot.get_foot_jacobian(q, foot_ids)
    
    # Current foot positions
    phi = robot.get_foot_positions(q, foot_ids)
    
    # Build KKT system
    tau = np.zeros(nv)
    tau[6:] = u
    
    nc = J.shape[0]
    H = np.block([
        [M,           J.T * dt],
        [J * dt,      1e-8 * np.eye(nc)]
    ])
    
    rhs = np.concatenate([
        (tau - h) * dt + M @ v,
        -(phi - foot_positions_target)
    ])
    
    sol = solve(H, rhs)
    v_next = sol[:nv]
    lambda_ = sol[nv:]
    
    # Integrate position
    q_next = pin.integrate(robot.model, q, v_next * dt)
    
    return np.concatenate([q_next, v_next]), lambda_
```

### Python Equilibrium Finder (CasADi)

```python
import casadi as ca

def find_equilibrium_casadi(robot, foot_contacts, q_guess):
    """
    Find static equilibrium using CasADi + IPOPT
    """
    nq = robot.nq
    nv = robot.nv
    nu = robot.nu
    nc = sum(foot_contacts) * 3  # 3 forces per contact
    
    # Decision variables
    opti = ca.Opti()
    q = opti.variable(nq)
    u = opti.variable(nu)
    lam = opti.variable(nc)
    
    # Objective: stay close to guess, minimize effort
    opti.minimize(
        ca.sumsqr(q - q_guess) + 
        1e-5 * ca.sumsqr(u) + 
        1e-5 * ca.sumsqr(lam)
    )
    
    # Constraints would use CasADi-compatible dynamics
    # (This requires CasADi-compatible pinocchio or manual dynamics)
    
    # Solve
    opti.solver('ipopt')
    sol = opti.solve()
    
    return sol.value(q), sol.value(u), sol.value(lam)
```

---

## Porting to C++

### Equivalent Libraries

| Julia | C++ Equivalent |
|-------|----------------|
| `RigidBodyDynamics.jl` | `pinocchio`, `RBDL`, `Drake` |
| `ForwardDiff.jl` | `CppAD`, `ADOL-C`, `casadi` |
| `Rotations.jl` | `Eigen::Quaternion`, `sophus` |
| `Ipopt.jl` | `Ipopt` (native C++) |
| `LinearAlgebra` | `Eigen` |

### Core Implementation (C++ + Pinocchio + Eigen)

```cpp
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

class QuadrupedBalance {
public:
    pinocchio::Model model;
    pinocchio::Data data;
    
    int nq, nv, nu;
    std::vector<pinocchio::FrameIndex> foot_frame_ids;
    
    QuadrupedBalance(const std::string& urdf_path) {
        pinocchio::urdf::buildModel(urdf_path, model);
        data = pinocchio::Data(model);
        
        nq = model.nq;
        nv = model.nv;
        nu = 12;  // 12 actuated joints
    }
    
    Eigen::VectorXd dynamics(const Eigen::VectorXd& x, 
                             const Eigen::VectorXd& u) {
        Eigen::VectorXd q = x.head(nq);
        Eigen::VectorXd v = x.tail(nv);
        
        // Mass matrix
        pinocchio::crba(model, data, q);
        Eigen::MatrixXd M = data.M;
        M.triangularView<Eigen::StrictlyLower>() = 
            M.transpose().triangularView<Eigen::StrictlyLower>();
        
        // Bias (Coriolis + gravity)
        Eigen::VectorXd h = pinocchio::nonLinearEffects(model, data, q, v);
        
        // Control mapping
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(nv);
        tau.tail(nu) = u;
        
        // Acceleration
        Eigen::VectorXd v_dot = M.ldlt().solve(tau - h);
        
        // Quaternion derivative (simplified)
        Eigen::VectorXd q_dot = v;  // Approximate for non-quaternion
        
        Eigen::VectorXd x_dot(nq + nv);
        x_dot << q_dot, v_dot;
        return x_dot;
    }
    
    Eigen::MatrixXd getFootJacobian(const Eigen::VectorXd& q) {
        pinocchio::computeJointJacobians(model, data, q);
        pinocchio::updateFramePlacements(model, data);
        
        int nc = foot_frame_ids.size() * 3;
        Eigen::MatrixXd J(nc, nv);
        
        for (size_t i = 0; i < foot_frame_ids.size(); ++i) {
            Eigen::MatrixXd Ji(6, nv);
            Ji.setZero();
            pinocchio::getFrameJacobian(model, data, foot_frame_ids[i],
                                        pinocchio::LOCAL_WORLD_ALIGNED, Ji);
            J.middleRows(i * 3, 3) = Ji.topRows(3);
        }
        return J;
    }
};


// Skew-symmetric matrix
Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}


// Left quaternion multiplication matrix
Eigen::Matrix4d L_quat(const Eigen::Vector4d& q) {
    double w = q(0), x = q(1), y = q(2), z = q(3);
    Eigen::Matrix4d L;
    L << w, -x, -y, -z,
         x,  w, -z,  y,
         y,  z,  w, -x,
         z, -y,  x,  w;
    return L;
}


// Matrix exponential discretization
void discretizeDynamics(const Eigen::MatrixXd& A,
                        const Eigen::MatrixXd& B,
                        const Eigen::MatrixXd& C,
                        double dt,
                        Eigen::MatrixXd& Ad,
                        Eigen::MatrixXd& Bd,
                        Eigen::MatrixXd& Cd) {
    int n = A.rows();
    int m = B.cols();
    int l = C.cols();
    
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + m + l, n + m + l);
    M.topLeftCorner(n, n) = A * dt;
    M.block(0, n, n, m) = B * dt;
    M.block(0, n + m, n, l) = C * dt;
    
    Eigen::MatrixXd M_exp = M.exp();
    
    Ad = M_exp.topLeftCorner(n, n);
    Bd = M_exp.block(0, n, n, m);
    Cd = M_exp.block(0, n + m, n, l);
}


// Constrained LQR
Eigen::MatrixXd maximalCoordinateLQR(
    const Eigen::MatrixXd& Ad,
    const Eigen::MatrixXd& Bd,
    const Eigen::MatrixXd& Cd,
    const Eigen::MatrixXd& D,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    int max_iter = 10000,
    double beta = 1e-5) {
    
    int n = Q.rows();
    int m = R.rows();
    int l = Cd.cols();
    int nc = D.rows();
    
    Eigen::MatrixXd P = Q;
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(m, n);
    Eigen::MatrixXd K_prev = K;
    
    for (int k = max_iter; k > 0; --k) {
        // Build KKT system
        Eigen::MatrixXd H(m + l + nc, m + l + nc);
        H.topLeftCorner(m, m) = R + Bd.transpose() * P * Bd;
        H.block(0, m, m, l) = Bd.transpose() * P * Cd;
        H.block(0, m + l, m, nc) = Bd.transpose() * D.transpose();
        H.block(m, 0, l, m) = Cd.transpose() * P * Bd;
        H.block(m, m, l, l) = beta * Eigen::MatrixXd::Identity(l, l) + 
                              Cd.transpose() * P * Cd;
        H.block(m, m + l, l, nc) = Cd.transpose() * D.transpose();
        H.block(m + l, 0, nc, m) = D * Bd;
        H.block(m + l, m, nc, l) = D * Cd;
        H.bottomRightCorner(nc, nc) = -beta * Eigen::MatrixXd::Identity(nc, nc);
        
        Eigen::MatrixXd b(m + l + nc, n);
        b.topRows(m) = Bd.transpose() * P * Ad;
        b.middleRows(m, l) = Cd.transpose() * P * Ad;
        b.bottomRows(nc) = D * Ad;
        
        // Solve
        Eigen::MatrixXd KLM = H.ldlt().solve(b);
        K = KLM.topRows(m);
        Eigen::MatrixXd L_ = KLM.middleRows(m, l);
        Eigen::MatrixXd M_mat = KLM.bottomRows(nc);
        
        // Update P
        Eigen::MatrixXd A_cl = Ad - Bd * K - Cd * L_;
        P = Q + K.transpose() * R * K + beta * L_.transpose() * L_ + 
            A_cl.transpose() * P * A_cl - M_mat.transpose() * D * A_cl;
        
        // Check convergence
        if ((K - K_prev).norm() < 1e-8) {
            std::cout << "Converged in " << (max_iter - k + 1) << " iterations\n";
            return K;
        }
        K_prev = K;
    }
    
    std::cerr << "Warning: Did not converge\n";
    return K;
}
```

---

## Porting to MATLAB

### Equivalent Functions

| Julia | MATLAB Equivalent |
|-------|-------------------|
| `RigidBodyDynamics.jl` | Robotics System Toolbox, `rigidBodyTree` |
| `ForwardDiff.jl` | Symbolic Math Toolbox, `jacobian()` |
| `Rotations.jl` | `quat2rotm`, `rotm2quat` |
| `Ipopt.jl` | `fmincon`, CasADi MATLAB |
| `expm` | `expm` (native) |

### Core Implementation (MATLAB)

```matlab
classdef QuadrupedBalance
    properties
        robot       % rigidBodyTree object
        nq, nv, nu
        foot_bodies
    end
    
    methods
        function obj = QuadrupedBalance(urdf_path)
            obj.robot = importrobot(urdf_path);
            obj.robot.DataFormat = 'column';
            obj.nq = obj.robot.NumBodies + 6;  % floating base
            obj.nv = obj.nq;
            obj.nu = 12;
        end
        
        function x_dot = dynamics(obj, x, u)
            q = x(1:obj.nq);
            v = x(obj.nq+1:end);
            
            M = massMatrix(obj.robot, q);
            h = velocityProduct(obj.robot, q, v) + ...
                gravityTorque(obj.robot, q);
            
            tau = [zeros(6,1); u];  % No actuation on floating base
            v_dot = M \ (tau - h);
            q_dot = v;  % Simplified
            
            x_dot = [q_dot; v_dot];
        end
        
        function J = getFootJacobian(obj, q)
            J = [];
            for i = 1:length(obj.foot_bodies)
                Ji = geometricJacobian(obj.robot, q, obj.foot_bodies{i});
                J = [J; Ji(4:6, :)];  % Position rows only
            end
        end
    end
end


function S = hat(v)
    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end


function L = L_quat(q)
    w = q(1); x = q(2); y = q(3); z = q(4);
    L = [w, -x, -y, -z;
         x,  w, -z,  y;
         y,  z,  w, -x;
         z, -y,  x,  w];
end


function [Ad, Bd, Cd] = discretize_dynamics(A, B, C, dt)
    n = size(A, 1);
    m = size(B, 2);
    l = size(C, 2);
    
    M = zeros(n + m + l);
    M(1:n, 1:n) = A * dt;
    M(1:n, n+1:n+m) = B * dt;
    M(1:n, n+m+1:end) = C * dt;
    
    M_exp = expm(M);
    
    Ad = M_exp(1:n, 1:n);
    Bd = M_exp(1:n, n+1:n+m);
    Cd = M_exp(1:n, n+m+1:end);
end


function K = maximal_coordinate_lqr(Ad, Bd, Cd, D, Q, R, max_iter, beta)
    if nargin < 7, max_iter = 10000; end
    if nargin < 8, beta = 1e-5; end
    
    n = size(Q, 1);
    m = size(R, 1);
    l = size(Cd, 2);
    nc = size(D, 1);
    
    P = Q;
    K_prev = zeros(m, n);
    
    for k = max_iter:-1:1
        % Build KKT system
        H = [R + Bd'*P*Bd,           Bd'*P*Cd,              Bd'*D';
             Cd'*P*Bd,     beta*eye(l) + Cd'*P*Cd,       Cd'*D';
             D*Bd,                    D*Cd,            -beta*eye(nc)];
        
        b = [Bd'*P*Ad; Cd'*P*Ad; D*Ad];
        
        KLM = H \ b;
        K = KLM(1:m, :);
        L_ = KLM(m+1:m+l, :);
        M_mat = KLM(m+l+1:end, :);
        
        A_cl = Ad - Bd*K - Cd*L_;
        P = Q + K'*R*K + beta*(L_'*L_) + A_cl'*P*A_cl - M_mat'*D*A_cl;
        
        if norm(K - K_prev) < 1e-8
            fprintf('Converged in %d iterations\n', max_iter - k + 1);
            return;
        end
        K_prev = K;
    end
    warning('Did not converge');
end
```

---

## Core Algorithm Pseudocode

This section provides language-agnostic pseudocode for the key algorithms.

### Equilibrium Finding

```
FUNCTION find_equilibrium(robot, foot_contacts, q_guess):
    
    # Decision variables: state x, control u, contact forces λ
    # x = [quaternion(4), position(3), joints(12), velocities(18)]
    
    DEFINE objective(x, u, λ):
        RETURN ||q - q_guess||² + α||u||² + β||λ||²
    
    DEFINE constraints(x, u, λ):
        # 1. Static equilibrium (zero acceleration)
        dynamics_residual = pinned_dynamics(x, u, λ)
        
        # 2. Foot positions match targets
        foot_residual = forward_kinematics(x) - foot_targets
        
        # 3. Quaternion normalization
        quat_norm = ||quaternion(x)|| - 1
        
        RETURN [dynamics_residual, foot_residual, quat_norm]
    
    SOLVE nonlinear_program:
        MINIMIZE objective(x, u, λ)
        SUBJECT TO constraints(x, u, λ) = 0
                    joint_limits_lower ≤ joints(x) ≤ joint_limits_upper
                    λ_z ≥ 0  # Unilateral contact (normal forces positive)
    
    RETURN x*, u*, λ*
```

### Linearization with Attitude Error

```
FUNCTION linearize_system(robot, x_eq, u_eq, λ_eq, foot_indices):
    
    # Compute Jacobians via automatic differentiation
    A_full = ∂(pinned_dynamics)/∂x  at (x_eq, u_eq, λ_eq)
    B_full = ∂(pinned_dynamics)/∂u  at (x_eq, u_eq, λ_eq)
    C_full = ∂(pinned_dynamics)/∂λ  at (x_eq, u_eq, λ_eq)
    
    # Contact constraint Jacobian
    D_full = ∂(foot_positions)/∂x at x_eq
    
    # Build attitude error Jacobian (37 -> 36 mapping)
    E = block_diagonal(
        quaternion_derivative_matrix(x_eq[1:4]),  # 4x3
        identity(33)                               # 33x33
    )
    
    # Transform to error coordinates
    A = E' * A_full * E   # 36x36
    B = E' * B_full       # 36x12
    C = E' * C_full       # 36x6
    D = D_full * E        # 6x36
    
    RETURN A, B, C, D
```

### Discretization

```
FUNCTION discretize(A, B, C, dt):
    n = rows(A)
    m = cols(B)
    l = cols(C)
    
    # Build augmented matrix
    M = zeros(n + m + l, n + m + l)
    M[1:n, 1:n] = A * dt
    M[1:n, n+1:n+m] = B * dt
    M[1:n, n+m+1:end] = C * dt
    
    # Matrix exponential
    M_exp = matrix_exponential(M)
    
    Ad = M_exp[1:n, 1:n]
    Bd = M_exp[1:n, n+1:n+m]
    Cd = M_exp[1:n, n+m+1:end]
    
    RETURN Ad, Bd, Cd
```

### Constrained LQR

```
FUNCTION maximal_coordinate_lqr(Ad, Bd, Cd, D, Q, R):
    n = size(Q)
    m = size(R)
    l = cols(Cd)
    nc = rows(D)
    β = 1e-5  # Regularization
    
    P = Q
    K = zeros(m, n)
    
    FOR k = max_iterations DOWN TO 1:
        # Build KKT system
        H = | R + Bd'*P*Bd        Bd'*P*Cd           Bd'*D'      |
            | Cd'*P*Bd      βI + Cd'*P*Cd         Cd'*D'      |
            | D*Bd               D*Cd              -βI         |
        
        b = | Bd'*P*Ad |
            | Cd'*P*Ad |
            | D*Ad     |
        
        [K, L, M] = solve(H, b)  # Partition solution
        
        # Riccati update
        A_cl = Ad - Bd*K - Cd*L
        P_new = Q + K'*R*K + β*L'*L + A_cl'*P*A_cl - M'*D*A_cl
        
        IF ||K - K_prev|| < tolerance:
            RETURN K
        
        P = P_new
        K_prev = K
    
    RETURN K
```

### Online Control Loop

```
FUNCTION control_loop(robot, x_eq, u_eq, K, foot_targets, dt):
    
    x = get_state_estimate()
    
    WHILE running:
        # Compute error state
        attitude_error = cayley_map_error(quaternion(x), quaternion(x_eq))
        position_error = position(x) - position(x_eq)
        joint_error = joints(x) - joints(x_eq)
        velocity_error = velocity(x)  # Assuming equilibrium has zero velocity
        
        δx = [attitude_error, position_error, joint_error, velocity_error]
        
        # Feedback control
        δu = -K * δx
        u = u_eq + δu
        
        # Apply torque limits
        u = clamp(u, torque_min, torque_max)
        
        # Send to robot
        send_torque_command(u)
        
        # Integrate (simulation) or wait (real robot)
        x, λ = semi_implicit_euler(robot, x, u, foot_targets, dt)
        
        wait(dt)
```

---

## Validation Checklist

Use this checklist to verify your port is working correctly:

### 1. Dynamics Validation

- [ ] Mass matrix is symmetric positive definite
- [ ] Dynamics bias includes gravity (robot falls without control)
- [ ] Forward dynamics matches inverse dynamics: `M * v_dot + h = tau`

### 2. Kinematics Validation

- [ ] FK matches URDF visualization
- [ ] Jacobian: `d(FK)/dt ≈ J * v` (numerical derivative check)
- [ ] Inverse kinematics converges for valid targets

### 3. Quaternion Math Validation

- [ ] `L(q1) * q2 = q1 ⊗ q2` (quaternion product)
- [ ] `||q|| = 1` maintained during integration
- [ ] Attitude error is zero for identical quaternions

### 4. Equilibrium Validation

- [ ] `pinned_dynamics(x_eq, u_eq, λ_eq) ≈ 0` (static)
- [ ] Foot positions match targets
- [ ] Joint angles within limits
- [ ] Contact forces have positive z-component

### 5. LQR Validation

- [ ] Closed-loop eigenvalues inside unit circle (discrete)
- [ ] Riccati iteration converges
- [ ] Small perturbations return to equilibrium in simulation

### 6. Integration Validation

- [ ] Energy approximately conserved without damping
- [ ] Contact constraints satisfied: `||φ(q)|| < tolerance`
- [ ] No interpenetration (z_foot ≥ 0)

### 7. Real-World Deployment

- [ ] Control loop runs faster than 1/dt
- [ ] State estimation noise doesn't cause instability
- [ ] Torque commands within motor limits
- [ ] Safe startup sequence (move to equilibrium slowly)

---

## Common Pitfalls

1. **Quaternion convention**: wxyz vs xyzw ordering varies by library
2. **Body vs world frame**: Velocities may be in different frames
3. **Joint ordering**: URDF order may differ from your convention
4. **Rotation direction**: Some libraries use passive vs active rotations
5. **Gravity sign**: Some libraries use +9.81, others -9.81
6. **Floating base DOF**: 6 or 7 depending on quaternion representation
7. **Jacobian frames**: LOCAL, WORLD, or LOCAL_WORLD_ALIGNED

---

## References

1. Pinocchio Documentation: https://stack-of-tasks.github.io/pinocchio/
2. Drake Documentation: https://drake.mit.edu/
3. Eigen Quick Reference: https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
4. CasADi User Guide: https://web.casadi.org/docs/
5. Joan Solà, "Quaternion kinematics for the error-state Kalman filter"
