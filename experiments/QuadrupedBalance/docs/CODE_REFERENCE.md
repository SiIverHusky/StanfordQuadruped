# QuadrupedBalance.jl - Complete Code Reference

This file contains all source code, constants, and notebook algorithms from the project, fully documented for standalone reference.

---

## Table of Contents

1. [Constants and Parameters](#constants-and-parameters)
2. [Utility Functions](#utility-functions)
3. [Rigid Body Model](#rigid-body-model)
4. [Quadruped Full Body Model](#quadruped-full-body-model)
5. [Forward Kinematics](#forward-kinematics)
6. [Centroidal Models](#centroidal-models)
7. [Equilibrium Finder (IPOPT)](#equilibrium-finder-ipopt)
8. [Maximal Coordinate LQR](#maximal-coordinate-lqr)
9. [Simulation Loop](#simulation-loop)
10. [Complete State/Control Dimensions](#complete-statecontrol-dimensions)

---

## Constants and Parameters

### Unitree A1 Kinematic Parameters

```julia
# From forward_kinematics.jl
const x_hip = 0.183      # Hip displacement from trunk frame (x-axis) [meters]
const y_hip = 0.047      # Hip displacement from trunk frame (y-axis) [meters]
const Δy_thigh = 0.08505 # Thigh displacement from hip frame [meters]
const l_limb = 0.2       # Link length (both thigh and calf) [meters]
```

### Motor ID Mappings

```julia
# From utils.jl
struct MotorIDs 
    FR_Hip::Int; FR_Thigh::Int; FR_Calf::Int
    FL_Hip::Int; FL_Thigh::Int; FL_Calf::Int 
    RR_Hip::Int; RR_Thigh::Int; RR_Calf::Int
    RL_Hip::Int; RL_Thigh::Int; RL_Calf::Int
end   

# C-style ordering (used by hardware interface)
const MotorIDs_c = MotorIDs(1,2,3,4,5,6,7,8,9,10,11,12)

# RigidBodyDynamics ordering (used internally)
const MotorIDs_rgb = MotorIDs(1,5,9,2,6,10,3,7,11,4,8,12)
```

### Quaternion Helper Matrix

```julia
# From utils.jl - Maps angular velocity to quaternion derivative
const H = [zeros(1,3); I]  # 4x3 matrix: [0 0 0; 1 0 0; 0 1 0; 0 0 1]
```

### Physical Constants

```julia
const g = [0.0, 0.0, -9.81]  # Gravity vector [m/s²]
```

---

## Utility Functions

### Quaternion Mathematics

```julia
# From utils.jl

"""
Left quaternion multiplication matrix.
For quaternions q1, q2: q1 ⊗ q2 = L(q1) * q2

Input: q = [w, x, y, z] (scalar-first convention)
Output: 4x4 matrix
"""
function L(Q)
    [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I + hat(Q[2:4])]
end

"""
Right quaternion multiplication matrix.
For quaternions q1, q2: q1 ⊗ q2 = R(q2) * q1

Input: q = [w, x, y, z] (scalar-first convention)
Output: 4x4 matrix
"""
function R(Q)
    [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I - hat(Q[2:4])]
end

"""
Skew-symmetric (hat) matrix from 3-vector.
hat(v) * u = v × u (cross product)

Input: v = [v1, v2, v3]
Output: 3x3 skew-symmetric matrix
"""
function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end

"""
Returns the G matrix for quaternion kinematics.
q̇ = 0.5 * G(q) * ω

Input: quat = [w, x, y, z]
Output: 4x3 matrix
"""
function get_G(quat)
    H = [zeros(3) I]'
    return L(quat) * H
end

"""
Quaternion differential matrix for error-state representation.
Maps 3-parameter attitude error to quaternion tangent space.

Input: quat = [w, x, y, z]
Output: 4x3 matrix
"""
function quaternion_differential(quat)
    return L(quat) * H
end

"""
Axis-angle to quaternion conversion (exponential map).
Used for attitude error computation.

Input: v = [θx, θy, θz] axis-angle vector
Output: q = [w, x, y, z] unit quaternion
"""
function ζ(v)
    v_norm = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    if v_norm != 0
        vec = sin(0.5*v_norm) * v/v_norm 
        scalar = cos(0.5*v_norm)
    else 
        scalar = 1 
        vec = zeros(3)
    end
    return [scalar; vec]
end

"""
Cayley map: 3-parameter to quaternion.
Alternative to axis-angle, simpler but less accurate for large rotations.

Input: ϕ = [ϕ1, ϕ2, ϕ3]
Output: q = [w, x, y, z] (approximately unit)
"""
function ρ(ϕ)
    q = 1/sqrt(1+norm(ϕ)) * [1; ϕ]
end

"""
Quaternion inverse (conjugate for unit quaternions).

Input: q = [w, x, y, z]
Output: q⁻¹ = [w, -x, -y, -z]
"""
function q_inv(q)
    return [q[1]; -q[2:end]]
end
```

### Integration Methods

```julia
# From utils.jl

"""
4th-order Runge-Kutta integration with zero-order hold on control.

Inputs:
    model: RobotDynamics.AbstractModel
    x: Current state
    u: Control input (held constant)
    h: Timestep
    
Output: x_next (state at t + h)
"""
function dynamics_rk4(model::RobotDynamics.AbstractModel, x, u, h)
    f1 = RobotDynamics.dynamics(model, x, u)
    f2 = RobotDynamics.dynamics(model, x + 0.5*h*f1, u)
    f3 = RobotDynamics.dynamics(model, x + 0.5*h*f2, u)
    f4 = RobotDynamics.dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end
```

### Jacobian Computation

```julia
# From utils.jl

"""
Compute discrete-time linearized dynamics using RK4.

Inputs:
    model: RobotDynamics.AbstractModel
    x: Linearization point (state)
    u: Linearization point (control)
    h: Timestep
    
Outputs:
    A: Discrete state matrix (∂x_{k+1}/∂x_k)
    B: Discrete control matrix (∂x_{k+1}/∂u_k)
"""
function dynamics_jacobians_discrete(model::RobotDynamics.AbstractModel, x, u, h)
    A = ForwardDiff.jacobian(t -> dynamics_rk4(model, t, u, h), x)
    B = ForwardDiff.jacobian(t -> dynamics_rk4(model, x, t, h), u)
    return A, B
end 

"""
Compute continuous-time linearized dynamics.

Inputs:
    model: RobotDynamics.AbstractModel
    x: Linearization point (state)
    u: Linearization point (control)
    
Outputs:
    A: Continuous state matrix (∂ẋ/∂x)
    B: Continuous control matrix (∂ẋ/∂u)
"""
function dynamics_jacobians(model::RobotDynamics.AbstractModel, x, u)
    A = ForwardDiff.jacobian(t -> dynamics(model, t, u), x)
    B = ForwardDiff.jacobian(t -> dynamics(model, x, t), u)
    return A, B
end 

"""
Compute linearized pinned dynamics (with contact forces).

Inputs:
    A1: AbstractQuadruped model
    xf: Linearization state
    uf: Linearization control
    λf: Linearization contact forces
    foot_indices: Which foot constraint dimensions are active
    
Outputs:
    A: ∂ẋ/∂x
    B: ∂ẋ/∂u
    C: ∂ẋ/∂λ
"""
function dynamics_jacobians(A1, xf, uf, λf, foot_indices)
    A = ForwardDiff.jacobian(t -> pinned_dynamics(A1, t, uf, λf, foot_indices), xf)
    B = ForwardDiff.jacobian(t -> pinned_dynamics(A1, xf, t, λf, foot_indices), uf)
    C = ForwardDiff.jacobian(t -> pinned_dynamics(A1, xf, uf, t, foot_indices), λf)
    return A, B, C
end 
```

### Motor Array Mapping

```julia
# From utils.jl

"""
Map motor commands between different ID conventions.

Inputs:
    cmd1: Motor command array in ids1 ordering
    ids1: Source MotorIDs struct
    ids2: Target MotorIDs struct
    
Output: cmd2 in ids2 ordering
"""
function mapMotorArrays(cmd1::AbstractVector{Float64}, ids1::MotorIDs, ids2::MotorIDs)
    cmd2 = @MVector zeros(length(cmd1))
    for motor in fieldnames(MotorIDs) 
        id1 = getfield(ids1, motor)
        id2 = getfield(ids2, motor)
        cmd2[id2] = cmd1[id1]
    end 
    cmd2 = SVector(cmd2)
    return cmd2
end 
```

---

## Rigid Body Model

```julia
# From rigidbodymodel.jl

""" 
RigidBodyModel{T}

A RigidBody model wrapper around RigidBodyDynamics.jl Mechanism.

Fields:
    mech: Mechanism from RigidBodyDynamics.jl
    statecache: Cached state for efficient computation
    dyncache: Cached dynamics result
    control_indices: Boolean vector indicating which DOFs are actuated
"""
struct RigidBodyModel{T} <: RobotDynamics.AbstractModel 
    mech::Mechanism{Float64}
    statecache::T
    dyncache::DynamicsResultCache{Float64}
    control_indices::Vector{Bool}
end 

"""
Constructor with explicit control indices.

Inputs:
    mech: Mechanism parsed from URDF
    control_indices: Boolean vector (true = actuated DOF)
"""
function RigidBodyModel(mech::Mechanism, control_indices::Vector{Bool})
    statecache = StateCache(mech)
    dyncache = DynamicsResultCache(mech)
    return RigidBodyModel{typeof(statecache)}(mech, statecache, dyncache, control_indices)
end 

"""
Constructor with all DOFs actuated (default).
"""
function RigidBodyModel(mech::Mechanism)
    statecache = StateCache(mech)
    dyncache = DynamicsResultCache(mech)
    control_indices = ones(Bool, num_positions(mech))
    return RigidBodyModel{typeof(statecache)}(mech, statecache, dyncache, control_indices)
end 

"""
State dimension: nq + nv (positions + velocities)
"""
RobotDynamics.state_dim(model::RigidBodyModel) = num_positions(model.mech) + num_velocities(model.mech)

"""
Control dimension: number of actuated DOFs
"""
RobotDynamics.control_dim(model::RigidBodyModel) = sum(model.control_indices)

"""
Compute continuous dynamics ẋ = f(x, u).

The state x = [q; v] where:
    q: Configuration (positions)
    v: Velocity

Inputs:
    model: RigidBodyModel
    x: State vector
    u: Control input (torques for actuated joints)
    
Output: ẋ = [q̇; v̇]
"""
function RobotDynamics.dynamics(model::RigidBodyModel, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2} 
    T = promote_type(T1, T2)
    state = model.statecache[T]
    res = model.dyncache[T]

    copyto!(state, x)
    
    # Build full torque vector (zeros for unactuated, u for actuated)
    τ = zeros(T, num_velocities(model.mech))
    τ[model.control_indices] .= u 
    
    dynamics!(res, state, τ)
    q̇ = res.q̇ 
    v̇ = res.v̇ 
    return [q̇; v̇]
end 

"""
Compute mass matrix M(q).

Input: x = [q; v] (only q is used)
Output: M (nv × nv symmetric positive definite matrix)
"""
function get_mass_matrix(model::RigidBodyModel, x::AbstractVector{T}) where T 
    s = model.statecache[T]
    copyto!(s, x)
    return mass_matrix(s) 
end 

"""
Compute dynamics bias C(q,v) (Coriolis + centrifugal + gravity).

Input: x = [q; v]
Output: C (nv-dimensional vector)

Note: The equation of motion is M*v̇ + C = τ + J'*λ
"""
function get_dynamics_bias(model::RigidBodyModel, x::Vector{Float64})
    s = model.statecache[Float64]
    copyto!(s, x)
    return dynamics_bias(s)
end 
```

---

## Quadruped Full Body Model

```julia
# From quadruped_fullbody.jl

"""
Abstract type for quadruped robots.
"""
abstract type AbstractQuadruped <: RobotDynamics.AbstractModel end 

"""
Full body quadruped model for Unitree A1.

State representation:
    x = [q; v]  (37-dimensional)
    
    q = [attitude (4×1 quaternion wxyz),
         position (3×1),
         hip_angles (4×1: FR, FL, RR, RL),
         thigh_angles (4×1),
         calf_angles (4×1)]
         
    v = [angular_velocity (3×1, body frame),
         linear_velocity (3×1, body frame),
         hip_velocities (4×1),
         thigh_velocities (4×1),
         calf_velocities (4×1)]
"""
struct UnitreeA1FullBody <: AbstractQuadruped
    rigidbody::RigidBodyModel 

    function UnitreeA1FullBody(mech::Mechanism) 
        # First 6 DOFs are floating base (unactuated)
        # Next 12 DOFs are joints (actuated)
        control_indices = Vector{Bool}([zeros(6); ones(12)])
        model = RigidBodyModel(mech, control_indices)
        new(model)
    end 
end 

"""
Forward dynamics wrapper.
"""
function dynamics(model::AbstractQuadruped, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2}
    return RobotDynamics.dynamics(model.rigidbody, x, u)
end 

"""
Mass matrix wrapper.
"""
function get_mass_matrix(model::AbstractQuadruped, x) 
    return get_mass_matrix(model.rigidbody, x)
end 

"""
Dynamics bias wrapper.
"""
function get_dynamics_bias(model::AbstractQuadruped, x)
    return get_dynamics_bias(model.rigidbody, x)
end 

RobotDynamics.state_dim(model::AbstractQuadruped) = RobotDynamics.state_dim(model.rigidbody)
RobotDynamics.control_dim(model::AbstractQuadruped) = RobotDynamics.control_dim(model.rigidbody)

"""
Dynamics with contact constraints (pinned feet).

This function computes ẋ with additional contact forces applied to keep
specified feet stationary.

Inputs:
    A1: AbstractQuadruped model
    x: State (37-dim)
    u: Control torques (12-dim)
    λ: Contact forces (n_contacts × 3 dim, stacked)
    foot_indices: Indices into foot position vector for active contacts
                  e.g., [1,2,3,10,11,12] for FR and RL feet
                  
Output: ẋ = [q̇; v̇] with contact forces included

The contact force contribution is: M⁻¹ * J' * λ
where J is the contact Jacobian in error-state coordinates.
"""
function pinned_dynamics(A1::QuadrupedBalance.AbstractQuadruped, 
                         x::AbstractVector{T1}, 
                         u::AbstractVector{T2}, 
                         λ::AbstractVector{T3}, 
                         foot_indices) where {T1, T2, T3}
    T = promote_type(typeof(x), typeof(u), typeof(λ))
    x = convert(T, x)
    u = convert(T, u)
    λ = convert(T, λ)
    T = promote_type(T1, T2, T3)
    
    # Build attitude error Jacobian for proper Jacobian computation
    # This maps from 37-dim state to 36-dim error state
    attitude_error_jacobian = blockdiag(
        sparse(0.5 * QuadrupedBalance.quaternion_differential(x[1:4])),
        sparse(Rotations.UnitQuaternion(x[1:4])),  # Rotation for position
        sparse(I(30))  # Joint angles and velocities
    )
    
    # Contact Jacobian in world frame, mapped to error coordinates
    J = QuadrupedBalance.dfk_world(x)[foot_indices, :] * attitude_error_jacobian
    J = J[:, 1:18]  # Only position-dependent part
    
    # Get mass matrix and unconstrained dynamics
    M = QuadrupedBalance.get_mass_matrix(A1, x)
    ẋ = QuadrupedBalance.dynamics(A1, x, u)
    
    # Add contact force contribution to accelerations
    ẋ[20:end] .= ẋ[20:end] .+ inv(M) * J' * λ
    
    return ẋ
end 
```

### Flywheel Variant

```julia
# From quadruped_flywheel.jl

"""
Unitree A1 with reaction wheel for angular momentum control.
Has 14 actuated DOFs (12 joints + 2 flywheel axes).
"""
struct UnitreeA1FlyWheel <: AbstractQuadruped
    rigidbody::RigidBodyModel

    function UnitreeA1FlyWheel(mech::Mechanism)
        # 6 unactuated (floating base) + 14 actuated (12 joints + 2 flywheel)
        control_indices = Vector{Bool}([zeros(6); ones(14)])
        model = RigidBodyModel(mech, control_indices)
        new(model)
    end 
end 

function RobotDynamics.dynamics(model::UnitreeA1FlyWheel, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2}
    return RobotDynamics.dynamics(model.rigidbody, x, u)
end 
```

---

## Forward Kinematics

```julia
# From forward_kinematics.jl

# Kinematic constants (defined at top of file)
const x_hip = 0.183      # Hip displacement from trunk (x-axis)
const y_hip = 0.047      # Hip displacement from trunk (y-axis)
const Δy_thigh = 0.08505 # Thigh offset from hip
const l_limb = 0.2       # Link length (thigh and calf equal)

"""
Analytical forward kinematics for A1 legs.

Given 12 joint angles, compute foot positions relative to body origin.

Input: q (12-dim) = [hip_FR, hip_FL, hip_RR, hip_RL,
                     thigh_FR, thigh_FL, thigh_RR, thigh_RL,
                     calf_FR, calf_FL, calf_RR, calf_RL]
                     
Output: p (12-dim) = [x_FR, y_FR, z_FR, x_FL, y_FL, z_FL,
                      x_RR, y_RR, z_RR, x_RL, y_RL, z_RL]
                      
Leg order: FR (Front-Right), FL (Front-Left), RR (Rear-Right), RL (Rear-Left)
"""
function fk(q::AbstractVector)
    # Extract joint angles for each leg
    q_FR = q[[1, 5, 9]]   # hip, thigh, calf
    q_FL = q[[2, 6, 10]]
    q_RR = q[[3, 7, 11]]
    q_RL = q[[4, 8, 12]]

    # Parametric FK for one leg
    # x_mir: +1 for front, -1 for rear
    # y_mir: -1 for right, +1 for left
    # θ: [hip, thigh, calf] angles
    fk_leg(x_mir, y_mir, θ) = [
        -l_limb * sin(θ[2] + θ[3]) - l_limb * sin(θ[2]) + x_hip * x_mir;
        y_hip * y_mir + Δy_thigh * y_mir * cos(θ[1]) + 
            l_limb * sin(θ[1]) * cos(θ[2] + θ[3]) + 
            l_limb * sin(θ[1]) * cos(θ[2]);
        Δy_thigh * y_mir * sin(θ[1]) - 
            l_limb * cos(θ[1]) * cos(θ[2] + θ[3]) - 
            l_limb * cos(θ[1]) * cos(θ[2])
    ]

    p = [fk_leg(1, -1, q_FR)...;   # Front-Right
         fk_leg(1, 1, q_FL)...;    # Front-Left
         fk_leg(-1, -1, q_RR)...;  # Rear-Right
         fk_leg(-1, 1, q_RL)...]   # Rear-Left
    return p
end

"""
Forward kinematics in world frame.

Given full state (37-dim), compute foot positions in world coordinates.

Input: x (37-dim) = [quaternion(4), position(3), joints(12), velocities(18)]
Output: p (12-dim) = world-frame foot positions

Process:
1. Compute body-frame foot positions from joint angles
2. Transform to world frame using body pose
"""
function fk_world(x::AbstractVector)
    q = x[8:19]  # Joint angles (indices 8-19)
    quat = Rotations.UnitQuaternion(x[1:4])  # Body orientation
    pos = x[5:7]  # Body position

    p_body = fk(q)  # Feet in body frame
    p_world = copy(p_body)

    # Transform each foot to world frame
    for i in 1:4 
        p_world[(i-1)*3 .+ (1:3)] = quat * p_body[(i-1)*3 .+ (1:3)] + pos 
    end 
    return p_world 
end 

"""
Kinematic Jacobian (body frame).

Input: q (12-dim joint angles)
Output: J (12×12 matrix) where ṗ = J * q̇
"""
function dfk(q::AbstractVector)
    return ForwardDiff.jacobian(t -> fk(t), q)
end 

"""
Kinematic Jacobian (world frame).

Input: x (37-dim full state)
Output: J (12×37 matrix) where ṗ_world = J * ẋ
"""
function dfk_world(x::AbstractVector)
    return ForwardDiff.jacobian(t -> fk_world(t), x)
end 

""" 
Inverse kinematics via gradient descent.

Given desired foot positions, find joint angles.

Inputs:
    p: Desired foot positions in body frame (12-dim)
    q_guess: Initial joint angle guess (12-dim)
    max_iter: Maximum iterations (default 100000)
    
Output: q (12-dim joint angles)

Algorithm: Gradient descent on ||p - fk(q)||² using Jacobian transpose
"""
function inv_kin(p::AbstractVector, q_guess::AbstractVector; max_iter = 100000)
    h = 0.1  # Step size
    p_now = fk(q_guess)
    res = norm(p - p_now)
    counter = 0 
    
    while res > 1e-7 
        # Jacobian transpose step (gradient descent)
        dq = dfk(q_guess)' * (p - p_now)
        q_guess = q_guess + dq * h 

        p_now = fk(q_guess)
        res = norm(p - p_now)
        
        if counter > max_iter
            println("did not converge")
            break
        end 
        counter += 1
    end 

    return q_guess 
end 
```

---

## Centroidal Models

### Basic Centroidal Model

```julia
# From centroidal_model.jl

"""
Simplified centroidal dynamics model.

Treats robot as single rigid body with forces applied at foot locations.

State (13-dim):
    x = [p (3), ṗ (3), q (4), ω (3)]
    
    p: Center of mass position
    ṗ: CoM velocity
    q: Body orientation (quaternion)
    ω: Angular velocity (body frame)
    
Control (12-dim):
    u = [f1 (3), f2 (3), f3 (3), f4 (3)]
    
    Forces at each foot (FR, FL, RR, RL)
"""
struct CentroidalModel <: RobotDynamics.AbstractModel
    J::Matrix      # Inertia tensor (3×3)
    m::Real        # Total mass
    p1::Vector     # FR foot position (world frame)
    p2::Vector     # FL foot position
    p3::Vector     # RR foot position
    p4::Vector     # RL foot position
end 

"""
Centroidal dynamics.

Equations:
    p̈ = (1/m) * Σfᵢ + g
    ω̇ = J⁻¹ * (R' * Στᵢ - ω × Jω)
    
where τᵢ = rᵢ × fᵢ (torque from each foot)
"""
function RobotDynamics.dynamics(model::CentroidalModel, x, u)
    g = [0.0, 0.0, -9.81]
    
    # Unpack state
    p = x[1:3]      # CoM position
    ṗ = x[4:6]      # CoM velocity
    q = x[7:10]     # Quaternion
    ω = x[11:13]    # Angular velocity

    # Rotation matrix from quaternion
    Q = H' * L(q) * R(q)' * H
    
    # Moment arms (foot positions relative to CoM)
    r1 = (model.p1 - p)
    r2 = (model.p2 - p)
    r3 = (model.p3 - p)
    r4 = (model.p4 - p)
    
    # Map individual foot forces to net force and torque
    # M * u gives [net_force; net_torque]
    M = [I(3)    I(3)    I(3)    I(3);
         hat(r1) hat(r2) hat(r3) hat(r4)]
    u_out = M * u
    f = u_out[1:3]   # Net force
    τ = u_out[4:6]   # Net torque (world frame)

    # Linear dynamics: p̈ = f/m + g
    p̈ = 1/model.m * f + g
    
    # Attitude kinematics: q̇ = 0.5 * L(q) * H * ω
    q̇ = 0.5 * L(q) * H * ω
    
    # Angular dynamics: Jω̇ = R'τ - ω × Jω
    damping = 0.0  
    ω_dot = model.J \ (Q' * τ - damping * ω - hat(ω) * model.J * ω)

    return [ṗ; p̈; q̇; ω_dot]
end

"""
Linearize centroidal model about (x, u).

Returns:
    A: ∂f/∂x (13×13)
    B: ∂f/∂u (13×12)
"""
function linearize(model, x, u)
    A = ForwardDiff.jacobian(t -> RobotDynamics.dynamics(model, t, u), x)
    B = ForwardDiff.jacobian(t -> RobotDynamics.dynamics(model, x, t), u)
    return A, B
end 
```

### Centroidal Pendulum (with Flywheel)

```julia
# From centroidal_pendulum.jl

"""
Centroidal model with flywheel for momentum control.

This model represents a quadruped balancing with feet pinned,
using a reaction wheel to control angular momentum.

State (15-dim):
    x = [p (3), ṗ (3), q (4), ω (3), ρ (2)]
    
    p, ṗ, q, ω: Same as CentroidalModel
    ρ: Flywheel angular momentum (x, y axes)
    
Control (2-dim):
    u = [τ_x, τ_y] - Flywheel torques
    
The flywheel creates reaction torques on the body.
"""
struct CentroidalPendulum <: RobotDynamics.AbstractModel
    J::Matrix                      # Body inertia
    m::Real                        # Mass
    foot_pos_world::Vector{Float64} # Foot positions in world frame (12-dim)
    foot_pos_body::Vector{Float64}  # Foot positions in body frame (12-dim)
    contacts::Vector{Bool}         # Which feet are in contact [FR, FL, RR, RL]
    
    function CentroidalPendulum(J, m, foot_pos_world, foot_pos_body, contacts)
        new(J, m, foot_pos_world, foot_pos_body, contacts)
    end
end 

"""
Constrained centroidal-pendulum dynamics.

The feet in contact are constrained to remain fixed.
This creates a DAE (differential-algebraic equation) system
solved via constraint force computation.
"""
function RobotDynamics.dynamics(model::CentroidalPendulum, x, u)
    g = [0.0, 0.0, 9.81]  # Note: +9.81 convention here
    
    # Unpack state
    p = x[1:3]
    ṗ = x[4:6]
    quat = x[7:10]
    ω = x[11:13]
    ρ = x[14:15]  # Flywheel angular momentum

    q = [p; quat]
    q̇ = [ṗ; ω]

    # Flywheel control input (2D to 3D mapping)
    u_f = [1.0 0.0;
           0.0 1.0;  
           0.0 0.0] * u 
    
    # Full flywheel momentum
    ρ_all = [1.0 0.0;
             0.0 1.0;
             0.0 0.0] * ρ

    # Mass matrix: [m*I, 0; 0, J]
    M = [model.m * I(3)  zeros(3, 3);
         zeros(3, 3)     model.J]
    
    # Dynamic bias: gravity and gyroscopic terms
    # The gyroscopic term includes flywheel: ω × (Jω + ρ)
    cor = [g..., (hat(ω) * (model.J * ω + ρ_all))...]
    
    # Contact constraint Jacobian and acceleration
    J = dcdq(model, q)
    d = ddcdq(model, q, q̇) * q̇
    
    # Solve KKT system for accelerations and constraint forces
    # [M, J'; J, εI] * [q̈; λ] = [-cor - reaction; -d]
    A = [M    J'; 
         J    I(6) * 1e-8]  # Small regularization
    res = [-cor - [0, 0, 0, u_f...]; 
           -d]
    out = A \ res 
    q̈ = out[1:6]

    p̈ = q̈[1:3]
    ω_dot = q̈[4:6]

    # Quaternion kinematics
    quat_dot = 0.5 * L(quat) * H * ω
    
    # Flywheel dynamics: ρ̇ = u
    ρ_dot = u

    return [ṗ; p̈; quat_dot; ω_dot; ρ_dot]
end

"""
Foot position constraint.

For feet in contact, compute: p_foot_world - p_foot_target = 0

Input: q = [p (3), quat (4)]
Output: Constraint violation vector
"""
function constraints(model::CentroidalPendulum, q::AbstractVector{T}) where T
    c = Array{T, 1}()
    p = q[1:3]
    quat = q[4:7]
    
    for i in 1:length(model.contacts)
        if model.contacts[i]
            r_w = model.foot_pos_world[(i-1)*3+1:(i-1)*3+3]  # Target
            r_b = model.foot_pos_body[(i-1)*3+1:(i-1)*3+3]   # Body frame
            append!(c, UnitQuaternion(quat) * r_b + p - r_w)
        end 
    end 

    return c 
end 

"""
Constraint Jacobian: ∂c/∂q in error coordinates.
"""
function dcdq(model::CentroidalPendulum, q)
    attitude_error_jacobian = SparseArrays.blockdiag(
        sparse(I(3)), 
        sparse(0.5 * quaternion_differential(q[4:7]))
    )
    return ForwardDiff.jacobian(t -> constraints(model, t), q) * attitude_error_jacobian
end 

"""
Constraint acceleration: d(J*q̇)/dt for Baumgarte stabilization.
"""
function ddcdq(model::CentroidalPendulum, q, q̇)
    attitude_error_jacobian = SparseArrays.blockdiag(
        sparse(I(3)), 
        sparse(0.5 * quaternion_differential(q[4:7]))
    )
    f(model, q) = dcdq(model, q) * q̇
    return ForwardDiff.jacobian(t -> f(model, t), q) * attitude_error_jacobian
end 

"""
Linearize pendulum model.
"""
function linearize(model, x, u)
    A = ForwardDiff.jacobian(t -> RobotDynamics.dynamics(model, t, u), x)
    B = ForwardDiff.jacobian(t -> RobotDynamics.dynamics(model, x, t), u)
    return A, B
end 
```

---

## Equilibrium Finder (IPOPT)

From `notebooks/Equilibrium finder (ipopt).ipynb`:

### IPOPT Interface Boilerplate

```julia
using MathOptInterface
using Ipopt
const MOI = MathOptInterface

"""
NLP problem structure for MathOptInterface.
"""
struct ProblemMOI <: MOI.AbstractNLPEvaluator
    n_nlp::Int           # Number of decision variables
    m_nlp::Int           # Number of constraints
    idx_ineq             # Indices of inequality constraints
    obj_grad::Bool       # Has objective gradient
    con_jac::Bool        # Has constraint Jacobian
    sparsity_jac         # Jacobian sparsity pattern
    sparsity_hess         # Hessian sparsity pattern (unused)
    primal_bounds         # Variable bounds
    constraint_bounds     # Constraint bounds
    hessian_lagrangian::Bool
end

"""
Generate dense sparsity pattern for Jacobian.
"""
function sparsity_jacobian(n, m)
    row = []
    col = []
    for cc in 1:n
        for rr in 1:m
            push!(row, rr)
            push!(col, cc)
        end
    end
    return collect(zip(row, col))
end

# Required MOI callbacks
MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x) = objective(x)

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    ForwardDiff.gradient!(grad_f, objective, x)
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator, g, x)
    constraint!(g, x)
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    ForwardDiff.jacobian!(reshape(jac, prob.m_nlp, prob.n_nlp), constraint!, zeros(prob.m_nlp), x)
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.sparsity_jac
```

### Solve Function

```julia
"""
Solve NLP using IPOPT.

Inputs:
    x0: Initial guess
    prob: ProblemMOI structure
    tol: Optimality tolerance
    c_tol: Constraint violation tolerance
    max_iter: Maximum iterations
    
Output: Solution vector z*
"""
function solve(x0, prob::MOI.AbstractNLPEvaluator;
        tol=1.0e-6, c_tol=1.0e-6, max_iter=10000)
    
    x_l, x_u = prob.primal_bounds
    c_l, c_u = prob.constraint_bounds

    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, prob, true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    x = MOI.add_variables(solver, prob.n_nlp)

    # Set variable bounds
    for i = 1:prob.n_nlp
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    return MOI.get(solver, MOI.VariablePrimal(), x)
end
```

### Problem Setup

```julia
# Load robot model
A1mech = parse_urdf("../src/a1/urdf/a1_light.urdf", floating=true, remove_fixed_tree_joints=false)
A1 = QuadrupedBalance.UnitreeA1FullBody(A1mech)

# Initial guess
x_guess = zeros(37)
x_guess[1] = 1.0  # Quaternion w = 1 (identity rotation)

# Set initial joint angles (standing pose)
x_guess[[1,5,9] .+ 7] = [0.0, 1.0, -2.0]   # FR leg
x_guess[[4,8,12] .+ 7] = [0.0, 1.0, -2.0]  # RL leg
x_guess[[2,6,10] .+ 7] = [0.0, 1.5, -2.5]  # FL leg (lifted)
x_guess[[3,7,11] .+ 7] = [0.0, 1.5, -2.5]  # RR leg (lifted)

# Specify contact pattern
foot_contacts = [1, 0, 0, 1]  # FR and RL in contact (diagonal stance)
foot_indices = []
for i in 1:length(foot_contacts)
    if foot_contacts[i] == 1
        append!(foot_indices, (i-1)*3 .+ (1:3))  # [1,2,3] for FR, [10,11,12] for RL
    end 
end 

# Get foot position constraints from initial guess
ϕ_constraint = QuadrupedBalance.fk_world(x_guess)[foot_indices, :]
x_guess[7] -= ϕ_constraint[3]  # Adjust height so foot touches ground
ϕ_constraint[3:3:length(foot_indices)] .= 0.0  # Feet at z=0

# Get joint limits from URDF
lim_upper = zeros(12)
lim_lower = zeros(12) 
for i in 4:15 
    lim_upper[i-3] = joints(A1mech)[i].position_bounds[1].upper 
    lim_lower[i-3] = joints(A1mech)[i].position_bounds[1].lower 
end 
```

### Objective and Constraints

```julia
"""
Objective: Minimize deviation from guess + regularization.

Decision vector: z = [x (37), u (12), λ (n_contacts*3)]
"""
function objective(z)
    α = 1e-5  # Control regularization
    f = 1e-5  # Force regularization
    
    q = z[1:19]      # Configuration
    u = z[38:49]     # Joint torques
    λ = z[50:end]    # Contact forces
    
    return (q - q_guess)' * (q - q_guess) + α * u' * u + f * λ' * λ
end 

"""
Constraints:
1. Static equilibrium: pinned_dynamics(x, u, λ) = 0
2. Foot positions: fk_world(x) = target
3. Quaternion normalization: ||q_att|| = 1
"""
function constraint!(c, z)
    x = z[1:37]      # State
    u = z[38:49]     # Control
    λ = z[50:end]    # Contact forces
    
    # Equilibrium constraint
    res_dyn = QB.pinned_dynamics(A1, x, u, λ, foot_indices)
    
    # Foot position constraint
    res_pos = QuadrupedBalance.fk_world(x)[foot_indices, :] - ϕ_constraint
    
    c[:] = [res_dyn;
            res_pos;
            norm(z[1:4])]  # Quaternion norm
end

"""
Variable bounds.
"""
function primal_bounds(n)
    x_l = ones(n) * -Inf 
    x_u = ones(n) * Inf 
    
    # Joint limits
    x_l[8:19] = lim_lower[1:12] 
    x_u[8:19] = lim_upper[1:12]
    
    # Contact forces: normal force ≥ 0 (unilateral contact)
    x_l[50:3:end] .= 0     # λ_z ≥ 0
    x_u[50:3:end] .= Inf    
    
    return x_l, x_u
end

"""
Constraint bounds (all equality except quaternion norm = 1).
"""
function constraint_bounds(m; idx_ineq=(1:0))
    c_l = zeros(m)
    c_l[end] = 1  # Quaternion norm = 1

    c_u = zeros(m)
    c_u[end] = 1
    
    return c_l, c_u
end
```

### Solve and Save

```julia
# Problem dimensions
n_nlp = 37 + 12 + 3 * sum(foot_contacts)  # state + control + forces
m_nlp = 37 + 3 * sum(foot_contacts) + 1   # dynamics + kinematics + quat

# Initial guess
z_guess = [x_guess..., zeros(12)..., zeros(3*sum(foot_contacts))...]

# Create and solve problem
prob = ProblemMOI(n_nlp, m_nlp, idx_ineq=(1:0)) 
z_sol = solve(z_guess, prob)

# Extract solution
x_eq = z_sol[1:37]
u_eq = z_sol[38:49]
λ_eq = z_sol[50:end]

# Save to TOML
data = Dict(
    "x_eq" => x_eq, 
    "u_eq" => u_eq,
    "λ_eq" => λ_eq
)
open("ipopt_eq_point.toml", "w") do io
    TOML.print(io, data)
end
```

---

## Maximal Coordinate LQR

From `notebooks/MaximalCoordinateLqr.ipynb`:

### The Constrained LQR Algorithm

```julia
"""
Maximal coordinate LQR with contact constraints.

Solves the infinite-horizon LQR problem:
    min Σ (x'Qx + u'Ru)
    s.t. x_{k+1} = Ad*x_k + Bd*u_k + Cd*λ_k
         D*x_{k+1} = 0  (contact constraint)

The constraint is handled by augmenting the Riccati equation.

Inputs:
    Ad: Discrete state matrix (36×36)
    Bd: Discrete control matrix (36×12)
    Cd: Discrete constraint force matrix (36×6)
    D: Contact constraint Jacobian (6×36)
    Q: State cost matrix (36×36)
    R: Control cost matrix (12×12)
    max_iter: Maximum Riccati iterations
    
Output:
    K: Feedback gain matrix (12×36)
    
Control law: u = u_eq - K * (x ⊖ x_eq)
where ⊖ denotes error-state computation
"""
function maximal_coordinate_lqr(Ad, Bd, Cd, D, Q, R, max_iter=10000)
    N = max_iter
    n = size(Q, 1)    # 36 (error state dim)
    m = size(R, 1)    # 12 (control dim)
    l = size(Cd, 2)   # 6 (constraint force dim)
    
    P_prev = zeros(n, n)
    K_prev = zeros(m, n)
    P = zeros(n, n)
    Qn = Q
    β = 1e-5  # Regularization parameter
    K = zeros(m, n)

    P_prev .= Qn
    K_prev .= K 
    
    for k = (N-1):-1:1
        # Build KKT system for optimal gains
        # Solves for [K; L; M] where:
        #   K: control gain
        #   L: constraint force gain  
        #   M: Lagrange multiplier gain
        H = [R + Bd'*P_prev*Bd     Bd'*P_prev*Cd           Bd'*D'; 
             Cd'*P_prev*Bd       β*1.0I(6)+Cd'*P_prev*Cd    Cd'*D'; 
             D*Bd                    D*Cd                   -β*1.0I(6)]

        b = [Bd'*P_prev*Ad; 
             Cd'*P_prev*Ad; 
             D*Ad]

        KLM = H \ b
        K = KLM[1:m, :]
        L_ = KLM[m+1:m+l, :]
        M = KLM[m+l+1:end, :]

        # Riccati update with constraint terms
        P .= Q + K'*R*K + β*L_'*L_ + 
             (Ad - Bd*K - Cd*L_)'*P_prev*(Ad - Bd*K - Cd*L_) - 
             M'*D*(Ad - Bd*K - Cd*L_)
        
        # Check convergence
        if norm(K_prev - K) < 1e-8
            println("Backward Ricatti converged in ", k, " iterations")
            return K
        end
        
        P_prev[:] .= P[:]
        K_prev[:] .= K[:]
    end

    return K
end
```

### Linearization and Discretization

```julia
# Load equilibrium point
data = TOML.parsefile("ipopt_eq_point.toml")
x_eq = data["x_eq"]
u_eq = data["u_eq"]
λ_eq = data["λ_eq"]

# Load robot model
urdfpath = joinpath(@__DIR__, "..", "src", "a1", "urdf", "a1.urdf")
A1mech = parse_urdf(urdfpath, floating=true, remove_fixed_tree_joints=false)
A1 = QuadrupedBalance.UnitreeA1FullBody(A1mech)

# Contact configuration
foot_contacts = [1, 0, 0, 1]  # FR, FL, RR, RL
foot_indices = []
for i in 1:length(foot_contacts)
    if foot_contacts[i] == 1
        append!(foot_indices, (i-1)*3 .+ (1:3))
    end 
end 

# Linearize about equilibrium
A, B, C = QuadrupedBalance.dynamics_jacobians(A1, x_eq, u_eq, λ_eq, foot_indices)

# Contact Jacobian
D_fd = QuadrupedBalance.dfk_world(x_eq)[foot_indices, :]

# Attitude error Jacobian (37 → 36 mapping)
# Converts quaternion (4-dim) to axis-angle error (3-dim)
attitude_error_jacobian = blockdiag(
    sparse(QuadrupedBalance.quaternion_differential(x_eq[1:4])),  # 4×3
    sparse(I(33))  # Rest unchanged
)

# Transform all matrices to error coordinates
D = D_fd * attitude_error_jacobian
A = attitude_error_jacobian' * A * attitude_error_jacobian  # 36×36
B = attitude_error_jacobian' * B                             # 36×12
C = attitude_error_jacobian' * C                             # 36×6

# Discretize using matrix exponential
n = 36   # Error state dim
m = 12   # Control dim 
n_c = 6  # Constraint dim (2 feet × 3 DOF)
h = 0.01 # Timestep [seconds]

# Build augmented matrix for discretization
O = [A B C]
O = [O; zeros(m + n_c, n + m + n_c)]

# Matrix exponential
O_exp = exp(O .* h)

# Extract discrete matrices
Ad = O_exp[1:n, 1:n]
Bd = O_exp[1:n, n+1:n+m]
Cd = O_exp[1:n, n+m+1:n+m+n_c]
```

### LQR Weight Selection

```julia
# State cost weights (Q diagonal)
Q_gains = zeros(36)

# Attitude error weights (indices 1-3)
Q_gains[1] = (1/deg2rad(10.0)^2)  # Roll: ±10° allowed
Q_gains[2] = (1/deg2rad(10.0)^2)  # Pitch
Q_gains[3] = (1/deg2rad(10.0)^2)  # Yaw

# Position error weights (indices 4-6)
Q_gains[4] = (1/0.01^2)   # X: ±1cm allowed
Q_gains[5] = (1/0.01^2)   # Y
Q_gains[6] = (1/1.5^2)    # Z: more tolerance (gravity direction)

# Joint angle weights (indices 7-18)
# Order: [hip_FR, hip_FL, hip_RR, hip_RL, thigh_FR, ..., calf_FR, ...]
# Higher weight on contacting legs (FR, RL = indices 1,4)
Q_gains[6 .+ [1,2,3,4]] .= 1 ./ ([deg2rad(15), deg2rad(1.5), deg2rad(1.5), deg2rad(15)]).^2  # Hips
Q_gains[6 .+ [5,6,7,8]] .= 1 ./ ([deg2rad(15), deg2rad(1.5), deg2rad(1.5), deg2rad(15)]).^2  # Thighs
Q_gains[6 .+ [9,10,11,12]] .= 1 ./ ([deg2rad(15), deg2rad(1.5), deg2rad(1.5), deg2rad(15)]).^2  # Calves

# Velocity weights (indices 19-36)
# Generally 1000× smaller than position weights (derivative relationship)
Q_gains[19] = Q_gains[1] / 1000  # Angular vel X
Q_gains[20] = Q_gains[2] / 1000  # Angular vel Y
Q_gains[21] = Q_gains[3] / 1000  # Angular vel Z
Q_gains[22] = Q_gains[4] / 1000  # Linear vel X
Q_gains[23] = Q_gains[5] / 1000  # Linear vel Y
Q_gains[24] = Q_gains[6] / 1000  # Linear vel Z
Q_gains[25:end] .= Q_gains[7:18] / 1000  # Joint velocities

# Control cost weights (R diagonal)
R_gains = zeros(12)
# Lower weight = cheaper to use that actuator
# Higher weight on non-contacting legs (FL, RR = indices 2,3)
R_gains[[1,2,3,4]] .= 1 ./ ([4.0, 1.0, 1.0, 4.0]).^2   # Hips
R_gains[[5,6,7,8]] .= 1 ./ ([4.0, 1.0, 1.0, 4.0]).^2   # Thighs
R_gains[[9,10,11,12]] .= 1 ./ ([4.0, 1.0, 1.0, 4.0]).^2  # Calves

# Build diagonal matrices
Q = sparse(Diagonal(Q_gains)) 
R = sparse(Diagonal(R_gains)) 

# Solve LQR
K = maximal_coordinate_lqr(Ad, Bd, Cd, D, Q, R)

# Save gain matrix
open("maximal_lqr_gain.txt", "w") do io
    writedlm(io, K)
end
```

---

## Simulation Loop

From `notebooks/BalanceSim.ipynb`:

### Semi-Implicit Euler Integrator

```julia
"""
Semi-implicit Euler integration with contact constraints.

This integrator:
1. Solves for velocities that satisfy constraints
2. Integrates positions using new velocities
3. Enforces contact constraint directly (no drift)

Inputs:
    A1: AbstractQuadruped model
    x: Current state (37-dim)
    u: Control torques (12-dim)
    ϕ_cons: Target foot positions for constrained feet
    foot_indices: Which foot DOFs are constrained
    h: Timestep
    
Outputs:
    xn: Next state
    λ: Constraint forces applied
"""
function semi_implicit_euler(A1::QB.AbstractQuadruped, 
                             x::AbstractVector, 
                             u::AbstractVector, 
                             ϕ_cons::AbstractVector, 
                             foot_indices, 
                             h)
    xn = copy(x)
    
    # Get dynamics quantities
    M = QB.get_mass_matrix(A1, xn)      # Mass matrix
    C_dyn = QB.get_dynamics_bias(A1, xn) # Coriolis + gravity
    
    # Build contact Jacobian in proper coordinates
    attitude_error_jacobian = blockdiag(
        sparse(0.5 * QuadrupedBalance.quaternion_differential(x[1:4])),
        sparse(Rotations.UnitQuaternion(x[1:4])),
        sparse(I(30))
    )
    J = QB.dfk_world(x)[foot_indices, :] * attitude_error_jacobian
    J = J[:, 1:18]  # Only velocity-dependent part
    
    # Current foot positions (constraint violation)
    ϕ = QB.fk_world(xn)[foot_indices]

    v_dim = num_velocities(A1.rigidbody.mech)  # 18
    p_dim = num_positions(A1.rigidbody.mech)   # 19
    
    # Joint damping for stability
    damp = 0.5
    
    # Build KKT system: [M, J'h; Jh, εI] * [v_next; λ] = [rhs1; rhs2]
    # This solves for velocities that:
    # 1. Satisfy M*v_dot + C = τ + J'λ
    # 2. Drive constraint violation ϕ to zero
    r = [([zeros(6); u] .- C_dyn - [zeros(6); damp*xn[p_dim+6+1:end]]) * h .+ M*xn[p_dim+1:end];
         -ϕ + ϕ_cons]
         
    H = zeros(v_dim + 6, v_dim + 6)  
    H[1:v_dim, 1:v_dim] .= M 
    H[1:v_dim, v_dim+1:end] .= J' * h 
    H[v_dim+1:end, 1:v_dim] .= J * h
    
    δ = H \ r

    v_next = δ[1:v_dim]
    λ = δ[v_dim+1:end]

    # Integrate positions
    # Body linear velocity is in body frame, convert to world
    rot = UnitQuaternion(xn[1:4])
    v_trans = rot * v_next[4:6]   
    xn[5:7] = xn[5:7] + v_trans * h
    
    # Joint positions
    xn[8:p_dim] .= xn[8:p_dim] + v_next[7:end] * h

    # Update quaternion
    q_dot = 0.5 * QB.get_G(x[1:4]) * v_next[1:3] * h
    xn[1:4] .= xn[1:4] + q_dot
    xn[1:4] .= xn[1:4] / norm(x[1:4])  # Renormalize
    
    # Update velocities
    xn[p_dim+1:end] .= v_next 
    
    return xn, λ
end 
```

### Main Simulation Loop

```julia
# Load equilibrium and gains
data = TOML.parsefile("ipopt_eq_point.toml")
x_eq = data["x_eq"]
u_eq = data["u_eq"]
K = readdlm("maximal_lqr_gain.txt", '\t', Float64, '\n')  # 36×12

# Simulation parameters
n = length(x_eq)  # 37
m = 12
h = 0.001         # 1 kHz control rate
tf = 5.0          # 5 second simulation
times = 0:h:tf

# Foot constraint positions (from equilibrium)
ϕ_cons = QB.fk_world(x_eq)[foot_indices]

# Pre-allocate storage
xs = zeros(length(times), n)
us = zeros(length(times)-1, m)
λs = zeros(length(times), 6)
x_errs = zeros(length(times)-1, n-1)

# Initial condition
xs[1, :] = copy(x_eq)
# Optional: add perturbation
# com_offset = [0.01, 0.0, 0.0]  # 1cm offset

# Main loop
for i in 1:length(times)-1
    ## Sensor readings (from state)
    encs = xs[i, 8:19]           # Joint encoders
    joint_vels = xs[i, 26:end]   # Joint velocities
    pos = xs[i, 5:7]             # Body position
    v = xs[i, 23:25]             # Body velocity
    ω = xs[i, 20:22]             # Body angular velocity
    quat_meas = UnitQuaternion(xs[i, 1:4])
    quat_eq = UnitQuaternion(x_eq[1:4])
    
    ## Compute error state (36-dim)
    # Position error
    x_errs[i, 4:6] = pos[1:3] - x_eq[5:7]
    
    # Attitude error via Cayley map
    θ_err = rotation_error(quat_meas, quat_eq, Rotations.CayleyMap()) 
    x_errs[i, 1:3] = θ_err 
    
    # Joint angle errors
    x_errs[i, 7:18] = encs - x_eq[8:19]
    
    # Velocity errors (equilibrium has zero velocity)
    x_errs[i, 19:21] = ω 
    x_errs[i, 22:24] = v 
    x_errs[i, 25:36] = joint_vels
    
    ## Feedback control: u = u_eq - K * error
    us[i, :] = -K * x_errs[i, :] + u_eq
    
    ## Integrate one step
    xs[i+1, :], λs[i+1, :] = semi_implicit_euler(A1, xs[i, :], us[i, :], ϕ_cons, foot_indices, h)
end
```

### Visualization

```julia
using MeshCat
using MeshCatMechanisms

# Create visualizer
vis = Visualizer() 
mvis = MechanismVisualizer(A1mech, URDFVisuals(urdfpath), vis)

# Create animation
q_anim = [xs[i, 1:19] for i in 1:length(times)-1]
animation = Animation(mvis, times[1:50:end-1], q_anim[1:50:end])  # Downsample for speed
setanimation!(mvis, animation)

# Open in browser
render(vis)
```

---

## Complete State/Control Dimensions

### State Vector Layout (37 dimensions)

| Index | Symbol | Description |
|-------|--------|-------------|
| 1-4 | `q_att` | Body orientation quaternion [w, x, y, z] |
| 5-7 | `p` | Body position [x, y, z] |
| 8 | `θ_hip_FR` | Front-right hip angle |
| 9 | `θ_hip_FL` | Front-left hip angle |
| 10 | `θ_hip_RR` | Rear-right hip angle |
| 11 | `θ_hip_RL` | Rear-left hip angle |
| 12 | `θ_thigh_FR` | Front-right thigh angle |
| 13 | `θ_thigh_FL` | Front-left thigh angle |
| 14 | `θ_thigh_RR` | Rear-right thigh angle |
| 15 | `θ_thigh_RL` | Rear-left thigh angle |
| 16 | `θ_calf_FR` | Front-right calf angle |
| 17 | `θ_calf_FL` | Front-left calf angle |
| 18 | `θ_calf_RR` | Rear-right calf angle |
| 19 | `θ_calf_RL` | Rear-left calf angle |
| 20-22 | `ω` | Body angular velocity [ωx, ωy, ωz] |
| 23-25 | `v` | Body linear velocity [vx, vy, vz] |
| 26 | `ω_hip_FR` | Front-right hip velocity |
| 27 | `ω_hip_FL` | Front-left hip velocity |
| 28 | `ω_hip_RR` | Rear-right hip velocity |
| 29 | `ω_hip_RL` | Rear-left hip velocity |
| 30 | `ω_thigh_FR` | Front-right thigh velocity |
| 31 | `ω_thigh_FL` | Front-left thigh velocity |
| 32 | `ω_thigh_RR` | Rear-right thigh velocity |
| 33 | `ω_thigh_RL` | Rear-left thigh velocity |
| 34 | `ω_calf_FR` | Front-right calf velocity |
| 35 | `ω_calf_FL` | Front-left calf velocity |
| 36 | `ω_calf_RR` | Rear-right calf velocity |
| 37 | `ω_calf_RL` | Rear-left calf velocity |

### Error State Vector Layout (36 dimensions)

Same as above, but indices 1-3 are axis-angle attitude error instead of quaternion.

| Index | Symbol | Description |
|-------|--------|-------------|
| 1-3 | `θ_err` | Attitude error [θx, θy, θz] (axis-angle) |
| 4-6 | `p_err` | Position error [Δx, Δy, Δz] |
| 7-18 | `θ_joints_err` | Joint angle errors (same order as state) |
| 19-21 | `ω_err` | Angular velocity error |
| 22-24 | `v_err` | Linear velocity error |
| 25-36 | `ω_joints_err` | Joint velocity errors |

### Control Vector Layout (12 dimensions)

| Index | Symbol | Description |
|-------|--------|-------------|
| 1 | `τ_hip_FR` | Front-right hip torque |
| 2 | `τ_hip_FL` | Front-left hip torque |
| 3 | `τ_hip_RR` | Rear-right hip torque |
| 4 | `τ_hip_RL` | Rear-left hip torque |
| 5 | `τ_thigh_FR` | Front-right thigh torque |
| 6 | `τ_thigh_FL` | Front-left thigh torque |
| 7 | `τ_thigh_RR` | Rear-right thigh torque |
| 8 | `τ_thigh_RL` | Rear-left thigh torque |
| 9 | `τ_calf_FR` | Front-right calf torque |
| 10 | `τ_calf_FL` | Front-left calf torque |
| 11 | `τ_calf_RR` | Rear-right calf torque |
| 12 | `τ_calf_RL` | Rear-left calf torque |

### Foot Index Mapping

For `foot_contacts = [FR, FL, RR, RL]`:

| Foot | Contact Index | Position Indices |
|------|---------------|------------------|
| FR (Front-Right) | 1 | 1, 2, 3 |
| FL (Front-Left) | 2 | 4, 5, 6 |
| RR (Rear-Right) | 3 | 7, 8, 9 |
| RL (Rear-Left) | 4 | 10, 11, 12 |

Example: `foot_contacts = [1, 0, 0, 1]` (diagonal stance) → `foot_indices = [1,2,3,10,11,12]`

---

## URDF Joint Order

The URDF defines joints in this order (matching RigidBodyDynamics parsing):

```
1. floating_base (6 DOF, unactuated)
2. FR_hip_joint
3. FL_hip_joint  
4. RR_hip_joint
5. RL_hip_joint
6. FR_thigh_joint
7. FL_thigh_joint
8. RR_thigh_joint
9. RL_thigh_joint
10. FR_calf_joint
11. FL_calf_joint
12. RR_calf_joint
13. RL_calf_joint
```

---

## File Format Specifications

### ipopt_eq_point.toml

```toml
x_eq = [1.0, 0.0, 0.0, 0.0, ...]  # 37 elements (state)
u_eq = [0.0, 0.0, 0.0, ...]       # 12 elements (torques)
λ_eq = [0.0, 0.0, 50.0, ...]      # 6 elements for 2 contacts (forces)
```

### maximal_lqr_gain.txt

Tab-delimited 12×36 matrix:
```
K[1,1]  K[1,2]  K[1,3]  ... K[1,36]
K[2,1]  K[2,2]  K[2,3]  ... K[2,36]
...
K[12,1] K[12,2] K[12,3] ... K[12,36]
```

Control law: `u = u_eq - K * error_state`
