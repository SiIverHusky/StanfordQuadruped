"""
Kinematics Validator for Diagonal Balance Experiment.

This tool validates the diagonal balance movements by:
1. Generating movement sequences
2. Running leg positions through inverse kinematics
3. Checking joint angles are within safe limits
4. Optionally checking PWM values

Usage:
    python -m experiments.diagonal_balance.validator
    python -m experiments.diagonal_balance.validator --pair FL_BR
    python -m experiments.diagonal_balance.validator --height 0.04 --verbose
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.diagonal_balance import config
from experiments.diagonal_balance.diagonal_movement import DiagonalMovementGenerator


class KinematicsValidator:
    """
    Validates movement sequences against kinematics and joint limits.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize the validator.
        
        Args:
            verbose: Print detailed information
        """
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        
        # Load configuration from pupper
        try:
            from pupper.Config import Configuration
            self.config = Configuration()
            self.config_loaded = True
        except ImportError:
            print("[Validator] Warning: Could not import pupper.Config, using defaults")
            self.config = None
            self.config_loaded = False
        
        # Load kinematics
        try:
            from pupper.Kinematics import four_legs_inverse_kinematics
            self.inverse_kinematics = four_legs_inverse_kinematics
            self.kinematics_loaded = True
        except ImportError:
            print("[Validator] Warning: Could not import Kinematics, skipping IK validation")
            self.inverse_kinematics = None
            self.kinematics_loaded = False
        
        # Load hardware interface for PWM calculation (optional)
        try:
            from pupper.HardwareInterface import angle_to_pwm
            from pupper.Config import ServoParams
            self.angle_to_pwm = angle_to_pwm
            self.servo_params = ServoParams()
            self.pwm_loaded = True
        except ImportError:
            print("[Validator] Warning: Could not import HardwareInterface, skipping PWM validation")
            self.angle_to_pwm = None
            self.pwm_loaded = False
        
        # Movement generator
        self.movement_gen = DiagonalMovementGenerator()
    
    def validate_position(self, position, frame_name=""):
        """
        Validate a single leg position matrix.
        
        Args:
            position: 3x4 numpy array of leg positions
            frame_name: Name for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        # Check position dimensions
        if position.shape != (3, 4):
            self.errors.append(f"{frame_name}: Invalid position shape {position.shape}, expected (3, 4)")
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
            self.errors.append(f"{frame_name}: Position contains NaN or Inf values")
            return False
        
        # Check reasonable position ranges (in meters)
        max_reach = 0.25  # Maximum reasonable leg reach
        for leg_idx in range(4):
            x, y, z = position[:, leg_idx]
            reach = np.sqrt(x**2 + y**2 + z**2)
            
            if reach > max_reach:
                self.errors.append(f"{frame_name}: Leg {leg_idx} reach {reach:.3f}m exceeds max {max_reach}m")
                valid = False
            
            # Z should typically be negative (below body)
            if z > 0.05:  # Allow some lift
                self.warnings.append(f"{frame_name}: Leg {leg_idx} Z={z:.3f}m is positive (lifted)")
        
        if self.verbose:
            print(f"  Position check: {frame_name} - {'PASS' if valid else 'FAIL'}")
        
        return valid
    
    def validate_joint_angles(self, angles, frame_name=""):
        """
        Validate joint angles against limits.
        
        Args:
            angles: 3x4 numpy array of joint angles (radians)
            frame_name: Name for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        if angles.shape != (3, 4):
            self.errors.append(f"{frame_name}: Invalid angles shape {angles.shape}, expected (3, 4)")
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            self.errors.append(f"{frame_name}: Angles contain NaN or Inf values")
            return False
        
        # Joint names for reporting
        joint_names = ["abduction", "hip", "knee"]
        
        for leg_idx in range(4):
            for joint_idx, joint_name in enumerate(joint_names):
                angle = angles[joint_idx, leg_idx]
                limits = config.JOINT_LIMITS[joint_name]
                
                if angle < limits["min"]:
                    self.errors.append(
                        f"{frame_name}: Leg {leg_idx} {joint_name} angle {np.degrees(angle):.1f}° "
                        f"< min {np.degrees(limits['min']):.1f}°"
                    )
                    valid = False
                
                elif angle > limits["max"]:
                    self.errors.append(
                        f"{frame_name}: Leg {leg_idx} {joint_name} angle {np.degrees(angle):.1f}° "
                        f"> max {np.degrees(limits['max']):.1f}°"
                    )
                    valid = False
        
        if self.verbose:
            print(f"  Joint angle check: {frame_name} - {'PASS' if valid else 'FAIL'}")
            for leg_idx in range(4):
                print(f"    Leg {leg_idx}: abd={np.degrees(angles[0,leg_idx]):.1f}° "
                      f"hip={np.degrees(angles[1,leg_idx]):.1f}° "
                      f"knee={np.degrees(angles[2,leg_idx]):.1f}°")
        
        return valid
    
    def validate_pwm(self, angles, frame_name=""):
        """
        Validate PWM values from joint angles.
        
        Args:
            angles: 3x4 numpy array of joint angles (radians)
            frame_name: Name for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        if not self.pwm_loaded:
            return True
        
        valid = True
        
        for leg_idx in range(4):
            for axis_idx in range(3):
                angle = angles[axis_idx, leg_idx]
                pwm = self.angle_to_pwm(angle, self.servo_params, axis_idx, leg_idx)
                
                if pwm < config.PWM_MIN:
                    self.errors.append(
                        f"{frame_name}: Leg {leg_idx} axis {axis_idx} PWM {pwm:.0f}µs "
                        f"< min {config.PWM_MIN}µs"
                    )
                    valid = False
                
                elif pwm > config.PWM_MAX:
                    self.errors.append(
                        f"{frame_name}: Leg {leg_idx} axis {axis_idx} PWM {pwm:.0f}µs "
                        f"> max {config.PWM_MAX}µs"
                    )
                    valid = False
        
        if self.verbose:
            print(f"  PWM check: {frame_name} - {'PASS' if valid else 'FAIL'}")
        
        return valid
    
    def validate_movement_sequence(self, pair="FR_BL", height=None):
        """
        Validate a complete movement sequence.
        
        Args:
            pair: Diagonal pair ("FR_BL" or "FL_BR")
            height: Lift height (default from config)
            
        Returns:
            True if all validations pass, False otherwise
        """
        height = height if height is not None else config.LIFT_HEIGHT
        
        print(f"\n{'='*60}")
        print(f"Validating Diagonal Balance Movement Sequence")
        print(f"{'='*60}")
        print(f"Diagonal pair: {pair}")
        print(f"Lift height:   {height} m")
        print(f"{'='*60}\n")
        
        # Clear previous errors
        self.errors = []
        self.warnings = []
        
        # Update config
        self.movement_gen.lift_height = height
        
        # Get leg positions for validation
        print("Generating movement sequence...")
        positions = self.movement_gen.get_leg_positions_for_validation(pair=pair)
        print(f"Generated {len(positions)} key frames\n")
        
        all_valid = True
        
        for i, position in enumerate(positions):
            frame_name = f"Frame {i+1}"
            print(f"Validating {frame_name}...")
            
            # Validate position
            if not self.validate_position(position, frame_name):
                all_valid = False
                continue
            
            # Run inverse kinematics
            if self.kinematics_loaded and self.config_loaded:
                try:
                    angles = self.inverse_kinematics(position, self.config)
                    
                    # Validate joint angles
                    if not self.validate_joint_angles(angles, frame_name):
                        all_valid = False
                    
                    # Validate PWM
                    if not self.validate_pwm(angles, frame_name):
                        all_valid = False
                        
                except Exception as e:
                    self.errors.append(f"{frame_name}: Inverse kinematics failed: {e}")
                    all_valid = False
            
            print()
        
        # Print summary
        print("="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  ⚠ {w}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  ✗ {e}")
            print(f"\n{'='*60}")
            print("VALIDATION FAILED")
            print("="*60)
            return False
        
        print(f"\n{'='*60}")
        print("VALIDATION PASSED")
        print("="*60)
        print("\nThe movement sequence is safe to execute.")
        print("Proceed with bench testing (robot secured):")
        print(f"  python -m experiments.diagonal_balance.diagonal_balancer --mode bench --pair {pair}")
        
        return True
    
    def run_sweep_test(self, heights=None):
        """
        Run validation across multiple lift heights.
        
        Args:
            heights: List of heights to test (default: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
            
        Returns:
            Dict of height -> pass/fail
        """
        if heights is None:
            heights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        
        print("\n" + "="*60)
        print("HEIGHT SWEEP TEST")
        print("="*60)
        
        results = {}
        
        for pair in ["FR_BL", "FL_BR"]:
            print(f"\nTesting pair: {pair}")
            for height in heights:
                # Temporarily suppress verbose output
                old_verbose = self.verbose
                self.verbose = False
                
                valid = self.validate_movement_sequence(pair=pair, height=height)
                results[(pair, height)] = valid
                
                self.verbose = old_verbose
                
                status = "✓ PASS" if valid else "✗ FAIL"
                print(f"  Height {height:.2f}m: {status}")
        
        print("\n" + "="*60)
        print("SWEEP SUMMARY")
        print("="*60)
        
        # Find maximum safe height for each pair
        for pair in ["FR_BL", "FL_BR"]:
            max_safe = 0
            for height in heights:
                if results.get((pair, height), False):
                    max_safe = height
            print(f"{pair}: Maximum safe height = {max_safe:.2f}m")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate diagonal balance kinematics"
    )
    parser.add_argument(
        "--pair",
        choices=["FR_BL", "FL_BR"],
        default=config.DIAGONAL_PAIR,
        help="Diagonal pair to validate (default: from config)"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=config.LIFT_HEIGHT,
        help="Lift height in meters (default: from config)"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run height sweep test"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = KinematicsValidator(verbose=args.verbose)
    
    if args.sweep:
        results = validator.run_sweep_test()
        # Return non-zero if any failed
        return 0 if all(results.values()) else 1
    else:
        valid = validator.validate_movement_sequence(pair=args.pair, height=args.height)
        return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
