# Diagonal Balance Experiment for Mini Pupper 2
# This module provides isolated code for balancing on 2 diagonal legs.
#
# Components:
#   - config.py:           Configuration and tuning parameters
#   - diagonal_movement.py: Movement generation for diagonal lift
#   - filters.py:          IIR and EKF filters for sensor fusion
#   - safety.py:           Safety manager and abort logic
#   - diagonal_balancer.py: Main control loop
#   - validator.py:        Dry-run kinematics validator
#
# Usage:
#   1. Run validator first:  python -m experiments.diagonal_balance.validator
#   2. Bench test:           python -m experiments.diagonal_balance.diagonal_balancer --mode bench
#   3. Physical test:        python -m experiments.diagonal_balance.diagonal_balancer --mode physical

__version__ = "0.1.0"
