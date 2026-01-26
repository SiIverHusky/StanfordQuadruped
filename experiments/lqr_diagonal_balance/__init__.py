"""
LQR-Based Diagonal Balance Experiment for Mini Pupper 2.

This experiment implements the balance control approach from:
"A New Balance Control Approach for Quadruped Robot with Diagonal Leg Support"

Key features:
1. LQR (Linear Quadratic Regulator) control for stable state feedback
2. 6-DOF model decomposed into three parts:
   - Part A: Posture control around diagonal axis (LQR)
   - Part B: Control along diagonal axis (PID)
   - Part C: Height control (PID)
3. VMC (Virtual Model Control) to map between diagonal joints and actual joints
4. Kalman filtering for noisy sensor data

Author: Experiment implementation
Date: 2026
"""

# Lazy imports to avoid circular dependencies and allow running submodules directly
__all__ = [
    'RobotParams',
    'LQRParams', 
    'DiagonalDynamics',
    'LQRController',
    'VirtualModelControl',
    'SixDOFDecomposition',
    'LQRDiagonalBalancer',
]

def __getattr__(name):
    """Lazy import to avoid import errors when running as module."""
    if name == 'RobotParams':
        from experiments.lqr_diagonal_balance.config import RobotParams
        return RobotParams
    elif name == 'LQRParams':
        from experiments.lqr_diagonal_balance.config import LQRParams
        return LQRParams
    elif name == 'DiagonalDynamics':
        from experiments.lqr_diagonal_balance.dynamics import DiagonalDynamics
        return DiagonalDynamics
    elif name == 'LQRController':
        from experiments.lqr_diagonal_balance.lqr_controller import LQRController
        return LQRController
    elif name == 'VirtualModelControl':
        from experiments.lqr_diagonal_balance.vmc import VirtualModelControl
        return VirtualModelControl
    elif name == 'SixDOFDecomposition':
        from experiments.lqr_diagonal_balance.decomposition import SixDOFDecomposition
        return SixDOFDecomposition
    elif name == 'LQRDiagonalBalancer':
        from experiments.lqr_diagonal_balance.controller import LQRDiagonalBalancer
        return LQRDiagonalBalancer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
