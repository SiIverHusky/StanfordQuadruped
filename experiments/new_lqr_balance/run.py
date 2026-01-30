#!/usr/bin/env python3
"""
Quick test script for the Reaction Wheel LQR Balance module.

Run with: python run.py [--mode simulation|bench|physical] [--duration 30]
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.new_lqr_balance.controller import ReactionWheelBalancer, OperatingMode, main


if __name__ == "__main__":
    main()
