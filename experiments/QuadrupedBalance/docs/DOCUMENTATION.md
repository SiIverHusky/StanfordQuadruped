# QuadrupedBalance.jl - Theory and Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Core Components](#core-components)
5. [Control Design Pipeline](#control-design-pipeline)
6. [Workflow Guide](#workflow-guide)
7. [File Reference](#file-reference)

---

## Project Overview

**QuadrupedBalance.jl** is a Julia package for quadruped robot balance control, specifically designed for the **Unitree A1** quadruped platform. The project implements:

- Full-body rigid body dynamics modeling
- Equilibrium pose finding via nonlinear optimization (IPOPT)
- **Maximal Coordinate LQR** control with contact constraints
- Constrained dynamics simulation with semi-implicit Euler integration
- Multiple model abstractions (full-body, centroidal, pendulum)

The primary goal is to stabilize a quadruped robot balancing on a subset of its feet (e.g., diagonal stance with only FR and RL legs in contact).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QuadrupedBalance.jl                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Robot Models   │    │  Control Design  │    │  Simulation   │  │
│  │                 │    │                  │    │               │  │
│  │ • RigidBodyModel│    │ • Equilibrium    │    │ • Semi-implicit│ │
│  │ • UnitreeA1Full │    │   Finder (IPOPT) │    │   Euler       │  │
│  │ • CentroidalModel│   │ • Maximal LQR    │    │ • Constrained │  │
│  │ • CentroidalPen │    │ • Linearization  │    │   Dynamics    │  │
│  └─────────────────┘    └──────────────────┘    └───────────────┘  │
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐                       │
│  │  Kinematics     │    │   Utilities      │                       │
│  │                 │    │                  │                       │
│  │ • Forward Kin.  │    │ • Quaternion Math│                       │
│  │ • Inverse Kin.  │    │ • RK4 Integration│                       │
│  │ • Jacobians     │    │ • Motor Mapping  │                       │
│  └─────────────────┘    └──────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### 1. State Representation

The full-body model uses **maximal coordinates** with the following state vector $\mathbf{x} \in \mathbb{R}^{37}$:

$$\mathbf{x} = \begin{bmatrix} \mathbf{q} \\ \mathbf{v} \end{bmatrix}$$

Where the configuration $\mathbf{q} \in \mathbb{R}^{19}$ is:

$$\mathbf{q} = \begin{bmatrix} 
\mathbf{q}_{\text{att}} & \text{(quaternion, 4×1)} \\
\mathbf{p}_{\text{pos}} & \text{(position, 3×1)} \\
\boldsymbol{\theta}_{\text{hip}} & \text{(hip angles, 4×1)} \\
\boldsymbol{\theta}_{\text{thigh}} & \text{(thigh angles, 4×1)} \\
\boldsymbol{\theta}_{\text{calf}} & \text{(calf angles, 4×1)}
\end{bmatrix}$$

And the velocity $\mathbf{v} \in \mathbb{R}^{18}$ is:

$$\mathbf{v} = \begin{bmatrix}
\boldsymbol{\omega} & \text{(angular velocity, 3×1)} \\
\dot{\mathbf{p}} & \text{(linear velocity, 3×1)} \\
\dot{\boldsymbol{\theta}}_{\text{joints}} & \text{(joint velocities, 12×1)}
\end{bmatrix}$$

**Leg ordering**: FR (Front-Right), FL (Front-Left), RR (Rear-Right), RL (Rear-Left)

### 2. Rigid Body Dynamics

The manipulator equation of motion:

$$\mathbf{M}(\mathbf{q})\dot{\mathbf{v}} + \mathbf{C}(\mathbf{q}, \mathbf{v}) = \mathbf{B}\boldsymbol{\tau} + \mathbf{J}^T \boldsymbol{\lambda}$$

Where:
- $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{18 \times 18}$ — Mass matrix
- $\mathbf{C}(\mathbf{q}, \mathbf{v}) \in \mathbb{R}^{18}$ — Coriolis, centrifugal, and gravity terms
- $\mathbf{B} \in \mathbb{R}^{18 \times 12}$ — Actuation selection matrix (joints only)
- $\boldsymbol{\tau} \in \mathbb{R}^{12}$ — Joint torques (control input)
- $\mathbf{J} \in \mathbb{R}^{n_c \times 18}$ — Contact Jacobian
- $\boldsymbol{\lambda} \in \mathbb{R}^{n_c}$ — Contact forces

### 3. Contact Constraints

For feet in contact with the ground, we enforce holonomic constraints:

$$\boldsymbol{\phi}(\mathbf{q}) = \mathbf{p}_{\text{foot}}^{\text{world}}(\mathbf{q}) - \mathbf{p}_{\text{contact}} = \mathbf{0}$$

The constraint Jacobian is:

$$\mathbf{J} = \frac{\partial \boldsymbol{\phi}}{\partial \mathbf{q}}$$

### 4. Pinned Dynamics (Constrained System)

The dynamics with contact constraints become:

$$\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}) + \begin{bmatrix} \mathbf{0} \\ \mathbf{M}^{-1}\mathbf{J}^T\boldsymbol{\lambda} \end{bmatrix}$$

Where the constraint forces $\boldsymbol{\lambda}$ ensure the feet remain fixed.

### 5. Quaternion Kinematics

The quaternion derivative follows:

$$\dot{\mathbf{q}}_{\text{att}} = \frac{1}{2} \mathbf{L}(\mathbf{q}_{\text{att}}) \mathbf{H} \boldsymbol{\omega}$$

Where $\mathbf{L}(\mathbf{q})$ is the left quaternion multiplication matrix and $\mathbf{H} = \begin{bmatrix} \mathbf{0}_{1\times3} \\ \mathbf{I}_3 \end{bmatrix}$.

### 6. Error State Representation

For control design, we use a reduced **error state** $\boldsymbol{\delta x} \in \mathbb{R}^{36}$ to avoid quaternion normalization issues:

$$\boldsymbol{\delta x} = \begin{bmatrix}
\boldsymbol{\theta}_{\text{err}} & \text{(attitude error via Cayley map, 3×1)} \\
\delta\mathbf{p} & \text{(position error, 3×1)} \\
\delta\boldsymbol{\theta}_{\text{joints}} & \text{(joint angle errors, 12×1)} \\
\delta\boldsymbol{\omega} & \text{(angular velocity error, 3×1)} \\
\delta\dot{\mathbf{p}} & \text{(velocity error, 3×1)} \\
\delta\dot{\boldsymbol{\theta}}_{\text{joints}} & \text{(joint velocity errors, 12×1)}
\end{bmatrix}$$

The attitude error Jacobian maps between quaternion and 3-parameter representations:

$$\mathbf{E} = \text{blockdiag}\left(\frac{1}{2}\mathbf{G}(\mathbf{q}_{\text{att}}), \mathbf{I}_{33}\right)$$

---

## Core Components

### 1. RigidBodyModel (`rigidbodymodel.jl`)

A wrapper around `RigidBodyDynamics.jl` mechanisms providing:

```julia
struct RigidBodyModel{T} <: RobotDynamics.AbstractModel 
    mech::Mechanism{Float64}      # URDF-parsed mechanism
    statecache::T                  # State cache for efficiency
    dyncache::DynamicsResultCache  # Dynamics result cache
    control_indices::Vector{Bool}  # Which DOFs are actuated
end
```

**Key Functions:**
- `dynamics(model, x, u)` — Compute $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$
- `get_mass_matrix(model, x)` — Compute $\mathbf{M}(\mathbf{q})$
- `get_dynamics_bias(model, x)` — Compute $\mathbf{C}(\mathbf{q}, \mathbf{v})$

### 2. UnitreeA1FullBody (`quadruped_fullbody.jl`)

Full-body model for the Unitree A1 with 12 actuated joints (hips, thighs, calves for each leg).

```julia
struct UnitreeA1FullBody <: AbstractQuadruped
    rigidbody::RigidBodyModel 
end
```

**Key Function:**
```julia
function pinned_dynamics(A1, x, u, λ, foot_indices)
    # Returns dynamics with contact constraint forces applied
    # foot_indices specifies which feet are pinned (e.g., [1,2,3,7,8,9] for FR and RR)
end
```

### 3. Forward Kinematics (`forward_kinematics.jl`)

Analytical forward kinematics for A1 legs with parameters:
- `x_hip = 0.183` m — Hip displacement from trunk (x-axis)
- `y_hip = 0.047` m — Hip displacement from trunk (y-axis)  
- `Δy_thigh = 0.08505` m — Thigh offset from hip
- `l_limb = 0.2` m — Link length (both thigh and calf)

**Key Functions:**
- `fk(q)` — Joint angles → foot positions in body frame (12-dim)
- `fk_world(x)` — Full state → foot positions in world frame (12-dim)
- `dfk(q)` — Kinematic Jacobian $\mathbf{J}_k \in \mathbb{R}^{12 \times 12}$
- `dfk_world(x)` — World-frame Jacobian $\mathbf{J}_k^w \in \mathbb{R}^{12 \times 37}$
- `inv_kin(p, q_guess)` — Inverse kinematics via gradient descent

### 4. CentroidalModel (`centroidal_model.jl`)

Simplified model treating the robot as a single rigid body with external forces at foot locations:

$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \dot{\mathbf{p}} \\ \mathbf{q}_{\text{att}} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{13}$$

**Dynamics:**

$$\ddot{\mathbf{p}} = \frac{1}{m}\sum_i \mathbf{f}_i + \mathbf{g}$$

$$\mathbf{J}\dot{\boldsymbol{\omega}} = \mathbf{R}^T\sum_i (\mathbf{r}_i \times \mathbf{f}_i) - \boldsymbol{\omega} \times \mathbf{J}\boldsymbol{\omega}$$

### 5. CentroidalPendulum (`centroidal_pendulum.jl`)

Extended centroidal model with **flywheel/reaction wheel** for angular momentum control:

$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \dot{\mathbf{p}} \\ \mathbf{q}_{\text{att}} \\ \boldsymbol{\omega} \\ \boldsymbol{\rho} \end{bmatrix} \in \mathbb{R}^{15}$$

Where $\boldsymbol{\rho}$ is the flywheel angular momentum. The control input drives flywheel acceleration:

$$\dot{\boldsymbol{\rho}} = \mathbf{u}_f$$

This creates a reaction torque on the body, enabling attitude control without ground reaction forces.

---

## Control Design Pipeline

### Step 1: Equilibrium Finding (IPOPT)

Find a static equilibrium pose $(\mathbf{x}^*, \mathbf{u}^*, \boldsymbol{\lambda}^*)$ satisfying:

**Optimization Problem:**

$$\min_{\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}} \quad \|\mathbf{q} - \mathbf{q}_{\text{guess}}\|^2 + \alpha\|\mathbf{u}\|^2 + \beta\|\boldsymbol{\lambda}\|^2$$

**Subject to:**
$$f(\mathbf{x}^*, \mathbf{u}^*) + \mathbf{J}^T\boldsymbol{\lambda}^* = \mathbf{0} \quad \text{(static equilibrium)}$$
$$\boldsymbol{\phi}(\mathbf{q}^*) = \boldsymbol{\phi}_{\text{target}} \quad \text{(foot positions)}$$
$$\|\mathbf{q}_{\text{att}}\| = 1 \quad \text{(quaternion normalization)}$$
$$\mathbf{q}_{\min} \leq \mathbf{q}_{\text{joints}} \leq \mathbf{q}_{\max} \quad \text{(joint limits)}$$
$$\lambda_z \geq 0 \quad \text{(unilateral contact, z-components)}$$

**Output:** `ipopt_eq_point.toml` containing `x_eq`, `u_eq`, `λ_eq`

### Step 2: Linearization

Linearize the constrained dynamics about the equilibrium:

$$\dot{\boldsymbol{\delta x}} = \mathbf{A}\boldsymbol{\delta x} + \mathbf{B}\boldsymbol{\delta u} + \mathbf{C}\boldsymbol{\delta\lambda}$$

With constraint:
$$\mathbf{D}\boldsymbol{\delta x} = \mathbf{0}$$

Where:
- $\mathbf{A} = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{(\mathbf{x}^*, \mathbf{u}^*, \boldsymbol{\lambda}^*)}$
- $\mathbf{B} = \frac{\partial f}{\partial \mathbf{u}}\bigg|_{(\mathbf{x}^*, \mathbf{u}^*, \boldsymbol{\lambda}^*)}$
- $\mathbf{C} = \frac{\partial f}{\partial \boldsymbol{\lambda}}\bigg|_{(\mathbf{x}^*, \mathbf{u}^*, \boldsymbol{\lambda}^*)}$
- $\mathbf{D} = \mathbf{J}_{\text{contact}}(\mathbf{x}^*) \cdot \mathbf{E}$ (contact Jacobian in error coordinates)

### Step 3: Discretization

Convert continuous dynamics to discrete-time using matrix exponential:

$$\begin{bmatrix} \mathbf{A}_d & \mathbf{B}_d & \mathbf{C}_d \\ \mathbf{0} & \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{I} \end{bmatrix} = \exp\left(h \cdot \begin{bmatrix} \mathbf{A} & \mathbf{B} & \mathbf{C} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} \end{bmatrix}\right)$$

### Step 4: Maximal Coordinate LQR

Solve the constrained LQR problem via backward Riccati iteration:

**Cost Function:**
$$J = \sum_{k=0}^{N-1} \left( \boldsymbol{\delta x}_k^T \mathbf{Q} \boldsymbol{\delta x}_k + \boldsymbol{\delta u}_k^T \mathbf{R} \boldsymbol{\delta u}_k \right)$$

**Constrained Dynamics:**
$$\boldsymbol{\delta x}_{k+1} = \mathbf{A}_d \boldsymbol{\delta x}_k + \mathbf{B}_d \boldsymbol{\delta u}_k + \mathbf{C}_d \boldsymbol{\delta\lambda}_k$$
$$\mathbf{D} \boldsymbol{\delta x}_{k+1} = \mathbf{0}$$

**Riccati Equation (with constraint):**

At each iteration, solve the KKT system:

$$\begin{bmatrix} 
\mathbf{R} + \mathbf{B}_d^T\mathbf{P}\mathbf{B}_d & \mathbf{B}_d^T\mathbf{P}\mathbf{C}_d & \mathbf{B}_d^T\mathbf{D}^T \\
\mathbf{C}_d^T\mathbf{P}\mathbf{B}_d & \beta\mathbf{I} + \mathbf{C}_d^T\mathbf{P}\mathbf{C}_d & \mathbf{C}_d^T\mathbf{D}^T \\
\mathbf{D}\mathbf{B}_d & \mathbf{D}\mathbf{C}_d & -\beta\mathbf{I}
\end{bmatrix}
\begin{bmatrix} \mathbf{K} \\ \mathbf{L} \\ \mathbf{M} \end{bmatrix}
= \begin{bmatrix} \mathbf{B}_d^T\mathbf{P}\mathbf{A}_d \\ \mathbf{C}_d^T\mathbf{P}\mathbf{A}_d \\ \mathbf{D}\mathbf{A}_d \end{bmatrix}$$

Where $\beta$ is a small regularization parameter.

**Output:** `maximal_lqr_gain.txt` containing the $36 \times 12$ gain matrix $\mathbf{K}$

### Step 5: Control Law

$$\mathbf{u} = \mathbf{u}^* - \mathbf{K}(\mathbf{x} \ominus \mathbf{x}^*)$$

Where $\ominus$ denotes the error-state computation (using Cayley map for attitude).

---

## Workflow Guide

### 1. Find Equilibrium Pose

Run `notebooks/Equilibrium finder (ipopt).ipynb`:

1. Load URDF model
2. Set initial pose guess
3. Define foot contact pattern (e.g., `foot_contacts = [1, 0, 0, 1]` for FR/RL diagonal stance)
4. Solve NLP for equilibrium
5. Save to `ipopt_eq_point.toml`

### 2. Compute LQR Gains

Run `notebooks/MaximalCoordinateLqr.ipynb`:

1. Load equilibrium point from TOML
2. Linearize dynamics about equilibrium
3. Apply attitude error Jacobian transformation
4. Discretize with timestep $h$ (default 0.01s)
5. Set Q and R weight matrices
6. Run backward Riccati
7. Save gains to `maximal_lqr_gain.txt`

### 3. Simulate

Run `notebooks/BalanceSim.ipynb`:

1. Load equilibrium and gains
2. Initialize state (optionally with perturbation)
3. Run semi-implicit Euler with contact constraints
4. Visualize with MeshCat

---

## File Reference

### Source Files (`src/`)

| File | Description |
|------|-------------|
| [QuadrupedBalance.jl](src/QuadrupedBalance.jl) | Main module, exports and includes |
| [rigidbodymodel.jl](src/rigidbodymodel.jl) | Generic rigid body dynamics wrapper |
| [quadruped_fullbody.jl](src/quadruped_fullbody.jl) | Full-body A1 model with pinned dynamics |
| [quadruped_flywheel.jl](src/quadruped_flywheel.jl) | A1 model with reaction wheel |
| [centroidal_model.jl](src/centroidal_model.jl) | Simplified centroidal dynamics |
| [centroidal_pendulum.jl](src/centroidal_pendulum.jl) | Centroidal + flywheel model |
| [forward_kinematics.jl](src/forward_kinematics.jl) | Analytical FK/IK for A1 |
| [utils.jl](src/utils.jl) | Math utilities (quaternions, integration, Jacobians) |

### Notebooks (`notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `Equilibrium finder (ipopt).ipynb` | Find static equilibrium via IPOPT |
| `MaximalCoordinateLqr.ipynb` | Compute constrained LQR gains |
| `BalanceSim.ipynb` | Constrained dynamics simulation |
| `UsefulFunctions.ipynb` | Function documentation and examples |
| `CentroidalPendulum.ipynb` | Centroidal + flywheel experiments |
| `FlyWheelSim.ipynb` | Reaction wheel simulation |
| `SysID.ipynb` | System identification utilities |

### Data Files (`notebooks/`)

| File | Contents |
|------|----------|
| `ipopt_eq_point.toml` | Equilibrium state/control/forces |
| `maximal_lqr_gain.txt` | $36 \times 12$ feedback gain matrix |

### URDF Models (`src/a1/urdf/`)

| File | Description |
|------|-------------|
| `a1.urdf` | Standard A1 model |
| `a1_light.urdf` | Lighter mass variant |
| `a1_rw.urdf` | A1 with reaction wheel |

---

## Key Dependencies

- **RigidBodyDynamics.jl** — Rigid body dynamics algorithms
- **RobotDynamics.jl** — Abstract model interface
- **ForwardDiff.jl** — Automatic differentiation for Jacobians
- **Rotations.jl** — 3D rotation representations
- **Ipopt.jl** — Interior point optimizer
- **MeshCat.jl** / **MeshCatMechanisms.jl** — 3D visualization
- **StaticArrays.jl** — Performance-optimized arrays

---

## Mathematical Notation Summary

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $\mathbf{x}$ | $37$ | Full state vector |
| $\mathbf{q}$ | $19$ | Configuration (quaternion + position + joints) |
| $\mathbf{v}$ | $18$ | Velocity |
| $\boldsymbol{\delta x}$ | $36$ | Error state (3-param attitude) |
| $\mathbf{u}$ | $12$ | Joint torques |
| $\boldsymbol{\lambda}$ | $n_c$ | Contact forces |
| $\mathbf{M}$ | $18 \times 18$ | Mass matrix |
| $\mathbf{J}$ | $n_c \times 18$ | Contact Jacobian |
| $\mathbf{K}$ | $36 \times 12$ | LQR gain matrix |
| $h$ | scalar | Timestep |
| $\mathbf{E}$ | $36 \times 37$ | Attitude error Jacobian |

---

## References

1. Featherstone, R. (2014). *Rigid Body Dynamics Algorithms*
2. Sola, J. (2017). "Quaternion kinematics for the error-state Kalman filter"
3. Di Carlo, J., et al. (2018). "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"
4. Bledt, G., et al. (2018). "MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot"
