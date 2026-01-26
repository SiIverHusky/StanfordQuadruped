#!/usr/bin/env python3
"""
Run the LQR Diagonal Balance Experiment.

This script is the entry point for running the experiment.
It supports different modes: simulation, bench test, and physical operation.

Usage:
    # Pure simulation (no hardware)
    python -m experiments.lqr_diagonal_balance.run --mode simulation
    
    # Bench test (robot secured on bench)
    python -m experiments.lqr_diagonal_balance.run --mode bench
    
    # Physical operation (robot free-standing)
    python -m experiments.lqr_diagonal_balance.run --mode physical
    
    # Run simulation with disturbance test
    python -m experiments.lqr_diagonal_balance.run --mode simulation --disturbance roll

    # Custom hold time
    python -m experiments.lqr_diagonal_balance.run --mode bench --hold-time 10.0

Based on the paper:
"A New Balance Control Approach for Quadruped Robot with Diagonal Leg Support"
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.lqr_diagonal_balance.config import Mode, HOLD_TIME
from experiments.lqr_diagonal_balance.controller import LQRDiagonalBalancer


def run_disturbance_tests():
    """Run a series of disturbance tests in simulation."""
    print("\n" + "="*60)
    print("DISTURBANCE TEST SUITE")
    print("="*60)
    print("Replicating tests from Section 5 of the paper")
    print("="*60 + "\n")
    
    balancer = LQRDiagonalBalancer(mode=Mode.SIMULATION)
    
    # Test 1: Roll angular velocity disturbance (like Figure 12)
    print("\n--- Test 1: Roll Angular Velocity Disturbance ---")
    for velocity in [1.0, 2.0, 3.0]:
        print(f"\nApplying {velocity} rad/s roll velocity...")
        history = balancer.run_simulation(
            duration=5.0,
            disturbance={'roll_velocity': velocity}
        )
    
    # Test 2: Yaw angular velocity disturbance (like Figure 13)
    print("\n--- Test 2: Yaw Angular Velocity Disturbance ---")
    for velocity in [1.0, 2.0, 3.0, 4.0]:
        print(f"\nApplying {velocity} rad/s yaw velocity...")
        # Yaw affects both legs
        history = balancer.run_simulation(
            duration=5.0,
            disturbance={'roll_velocity': velocity * 0.5, 'pitch_velocity': velocity * 0.5}
        )
    
    # Test 3: Lateral velocity disturbance (like Figure 14)
    print("\n--- Test 3: Lateral Velocity Disturbance ---")
    for velocity in [0.1, 0.2]:
        print(f"\nApplying {velocity} m/s lateral velocity...")
        # Approximate as angular velocity
        history = balancer.run_simulation(
            duration=5.0,
            disturbance={'roll_velocity': velocity * 5}
        )
    
    print("\n" + "="*60)
    print("DISTURBANCE TESTS COMPLETE")
    print("="*60)


def analyze_lqr_design():
    """Analyze and print the LQR controller design."""
    print("\n" + "="*60)
    print("LQR CONTROLLER ANALYSIS")
    print("="*60 + "\n")
    
    from experiments.lqr_diagonal_balance.dynamics import DiagonalDynamics
    from experiments.lqr_diagonal_balance.lqr_controller import LQRController
    from experiments.lqr_diagonal_balance.config import LQRParams
    
    # Create dynamics and controller
    dynamics = DiagonalDynamics()
    controller = LQRController(dynamics)
    
    # Get stability info
    info = controller.get_stability_info()
    
    print("State Vector: [θ_F, θ_F_dot, θ_H, θ_H_dot, θ_body, θ_body_dot]")
    print("\nState Cost Matrix Q (diagonal):")
    print(f"  {np.diag(LQRParams.Q)}")
    
    print("\nControl Cost Matrix R (diagonal):")
    print(f"  {np.diag(LQRParams.R)}")
    
    print("\nLinearized System Matrices:")
    print(f"A matrix shape: {controller.A.shape}")
    print(f"B matrix shape: {controller.B.shape}")
    
    print("\nLQR Gain Matrix K:")
    print(f"  Shape: {controller.K.shape}")
    print(f"  K = {controller.K}")
    
    print("\nClosed-Loop Analysis:")
    print(f"  Stable: {info['stable']}")
    print(f"  Dominant eigenvalue: {info['dominant_eigenvalue']:.4f}")
    print(f"  Approximate settling time: {info['settling_time_approx']:.2f} s")
    
    print("\nAll closed-loop eigenvalues:")
    for i, ev in enumerate(info['eigenvalues']):
        print(f"  λ_{i+1} = {ev.real:+.4f} + {ev.imag:+.4f}j")
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LQR Diagonal Balance Experiment for Mini Pupper 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Simulation:  python -m experiments.lqr_diagonal_balance.run --mode simulation
  Bench test:  python -m experiments.lqr_diagonal_balance.run --mode bench
  Full test:   python -m experiments.lqr_diagonal_balance.run --mode physical
  
  Analysis:    python -m experiments.lqr_diagonal_balance.run --analyze
  Disturbance: python -m experiments.lqr_diagonal_balance.run --disturbance-tests
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['simulation', 'bench', 'physical'],
        default='simulation',
        help='Operating mode (default: simulation)'
    )
    
    parser.add_argument(
        '--hold-time', '-t',
        type=float,
        default=HOLD_TIME,
        help=f'Time to hold diagonal stance in seconds (default: {HOLD_TIME})'
    )
    
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Print LQR controller analysis and exit'
    )
    
    parser.add_argument(
        '--disturbance-tests', '-d',
        action='store_true',
        help='Run disturbance test suite in simulation'
    )
    
    parser.add_argument(
        '--support-pair', '-p',
        choices=['FR_BL', 'FL_BR'],
        default='FL_BR',
        help='Which diagonal pair should support (default: FL_BR)'
    )
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.analyze:
        analyze_lqr_design()
        return 0
    
    if args.disturbance_tests:
        run_disturbance_tests()
        return 0
    
    # Map mode string to Mode enum
    mode_map = {
        'simulation': Mode.SIMULATION,
        'bench': Mode.BENCH,
        'physical': Mode.PHYSICAL,
    }
    
    mode = mode_map[args.mode]
    
    # Safety confirmation for physical mode
    if mode == Mode.PHYSICAL:
        print("\n" + "!"*60)
        print("WARNING: PHYSICAL MODE SELECTED")
        print("The robot will move freely. Ensure:")
        print("  1. Clear area around the robot")
        print("  2. Emergency stop ready")
        print("  3. Robot is on a flat, non-slip surface")
        print("!"*60)
        
        response = input("\nType 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Aborted.")
            return 1
    
    # Create and run balancer
    print(f"\nStarting LQR Diagonal Balance in {args.mode} mode...")
    
    # Update support pair in config if specified
    if args.support_pair:
        from experiments.lqr_diagonal_balance import config
        config.SUPPORT_PAIR = args.support_pair
    
    balancer = LQRDiagonalBalancer(mode=mode)
    
    try:
        if mode == Mode.SIMULATION:
            # Run simulation with optional disturbance
            balancer.run_simulation(duration=args.hold_time + 2.0)
        else:
            # Run on hardware
            balancer.run(hold_time=args.hold_time)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        balancer.stop()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
