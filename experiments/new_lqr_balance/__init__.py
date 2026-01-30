"""
New LQR Balance Controller - Reaction Wheel Inverted Pendulum Model

This module implements a two-leg diagonal balance controller using
the reaction-wheel inverted pendulum model:
- Robot stands on two diagonal legs; other two legs lifted
- Body + lifted legs → reaction wheel
- Supporting legs → pendulum

State vector: X = [θ_rel, θ_body, ω_rel, ω_body]^T
where:
    θ_rel: relative angle between body and supporting legs
    θ_body: absolute body inclination angle
    ω_rel, ω_body: angular velocities

Control: State-feedback LQR with cooperative torque distribution
"""

from experiments.new_lqr_balance.config import (
    RobotParams,
    ControlParams,
    SafetyParams,
    SensorParams,
    CONTROL_DT,
    SUPPORT_PAIR,
)
from experiments.new_lqr_balance.state_estimator import StateEstimator
from experiments.new_lqr_balance.lqr_controller import ReactionWheelLQR
from experiments.new_lqr_balance.torque_converter import TorqueToPositionConverter
from experiments.new_lqr_balance.controller import ReactionWheelBalancer

__all__ = [
    'RobotParams',
    'ControlParams',
    'SafetyParams',
    'SensorParams',
    'StateEstimator',
    'ReactionWheelLQR',
    'TorqueToPositionConverter',
    'ReactionWheelBalancer',
]
