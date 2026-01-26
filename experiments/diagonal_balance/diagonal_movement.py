"""
Diagonal Movement Generator for Mini Pupper 2.

Generates Movement sequences for:
1. CoM shift - shifts center of mass over supporting diagonal
2. Diagonal lift - raises two diagonal legs while other two support
3. Balance correction - adjusts body attitude to maintain balance
4. Return to stand - safely returns to default stance

This module does NOT modify any original StanfordQuadruped files.
It creates Movement-compatible objects that work with MovementScheme.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.MovementScheme import Movements
from experiments.diagonal_balance import config


class DiagonalMovementGenerator:
    """Generates Movement sequences for diagonal leg balancing."""
    
    def __init__(self):
        """Initialize the movement generator with configuration."""
        self.dt = config.DT
        self.default_stance = config.DEFAULT_STANCE
        self.lift_height = config.LIFT_HEIGHT
        self.lift_time = config.LIFT_TIME
        self.hold_time = config.HOLD_TIME
        self.return_time = config.RETURN_TIME
        
        # Movement library to accumulate movements
        self.MovementLib = []
        
        # Caps for safety (matching MovementGroup.py)
        self.rowcap = 25
        self.pitchcap = 20
        self.legliftcap = 0.06
    
    def cap_limit(self, max_val, min_val, value):
        """Clamp value between min and max for motor safety."""
        return max(min_val, min(max_val, value))
    
    def clear_movements(self):
        """Clear the movement library."""
        self.MovementLib = []
    
    def get_diagonal_config(self, pair="FR_BL"):
        """
        Get leg configuration for a diagonal pair.
        
        Args:
            pair: "FR_BL" or "FL_BR"
            
        Returns:
            dict with 'lift' and 'support' leg indices
        """
        return config.DIAGONAL_PAIRS.get(pair, config.DIAGONAL_PAIRS["FR_BL"])
    
    def com_shift(self, pair="FR_BL", shift_x=None, shift_y=None, time_shift=None):
        """
        Generate a center-of-mass shift movement.
        
        Shifts the body weight toward the supporting diagonal before lifting.
        
        Args:
            pair: "FR_BL" or "FL_BR" - which diagonal to lift
            shift_x: X shift distance (m), default from config
            shift_y: Y shift distance (m), default from config
            time_shift: Transition time (s), default from config
            
        Returns:
            Updated MovementLib
        """
        shift_x = shift_x if shift_x is not None else config.COM_SHIFT_X
        shift_y = shift_y if shift_y is not None else config.COM_SHIFT_Y
        time_shift = time_shift if time_shift is not None else config.COM_SHIFT_TIME
        
        diagonal_config = self.get_diagonal_config(pair)
        lift_legs = diagonal_config["lift"]
        support_legs = diagonal_config["support"]
        
        interval = max(1, int(time_shift / self.dt))
        
        # Calculate shift direction based on which diagonal supports
        # For FR_BL lift (support FL+BR): shift toward FL+BR center
        # For FL_BR lift (support FR+BL): shift toward FR+BL center
        if pair == "FR_BL":
            # Support is FL (1) and BR (2) - shift left and slightly back
            x_dir = -1
            y_dir = 1
        else:
            # Support is FR (0) and BL (3) - shift right and slightly forward
            x_dir = 1
            y_dir = -1
        
        # Build leg positions with CoM shift
        dance_all_legs = []
        for leg_idx in range(4):
            base = self.default_stance[leg_idx][0].copy()
            # Shift all legs to move body (inverse of foot movement)
            shifted = [
                base[0] - x_dir * shift_x,
                base[1] - y_dir * shift_y,
                base[2]
            ]
            # Keep as sequence for Movements
            dance_all_legs.append([[shifted[0], shifted[1], shifted[2]],
                                   [shifted[0], shifted[1], shifted[2]]])
        
        dance_scheme = Movements('com_shift')
        dance_speed = [[0, 0, 0]]
        dance_attitude = [[0, 0, 0]]
        
        dance_scheme.setInterpolationNumber(interval)
        dance_scheme.setTransitionTic(interval)
        dance_scheme.setAllSequence(dance_all_legs, dance_speed, dance_attitude)
        
        self.MovementLib.append(dance_scheme)
        return self.MovementLib
    
    def diagonal_lift(self, pair="FR_BL", height=None, time_lift=None, time_hold=None):
        """
        Generate a diagonal leg lift movement.
        
        Lifts two diagonal legs while the other two support the body.
        Includes body attitude compensation to keep balance.
        
        Args:
            pair: "FR_BL" or "FL_BR" - which diagonal to lift
            height: Lift height (m), default from config
            time_lift: Time to complete lift (s), default from config
            time_hold: Time to hold position (s), default from config
            
        Returns:
            Updated MovementLib
        """
        height = height if height is not None else self.lift_height
        time_lift = time_lift if time_lift is not None else self.lift_time
        time_hold = time_hold if time_hold is not None else self.hold_time
        
        # Clamp height for safety
        height = self.cap_limit(self.legliftcap, 0, height)
        
        diagonal_config = self.get_diagonal_config(pair)
        lift_legs = diagonal_config["lift"]
        support_legs = diagonal_config["support"]
        
        # Longer transition time for gradual lift - this is key for stability
        interval_lift = max(1, int(time_lift / self.dt))
        interval_hold = max(1, int(time_hold / self.dt))
        
        # Build leg positions
        # Lifted legs: raise Z by height, optionally move forward/outward
        # Support legs: lower slightly to increase ground force
        dance_all_legs = []
        
        for leg_idx in range(4):
            base = self.default_stance[leg_idx][0].copy()
            
            if leg_idx in lift_legs:
                # Lift this leg - final lifted position
                lifted = [
                    base[0] + 0.02,  # Move slightly forward
                    base[1],
                    base[2] + height  # Raise (more positive Z)
                ]
                # Single keyframe: MovementScheme will interpolate from current position
                dance_all_legs.append([[lifted[0], lifted[1], lifted[2]]])
            else:
                # Support leg - push down slightly for stability
                support_drop = 0.015  # Increased drop to increase contact force
                lowered = [
                    base[0],
                    base[1],
                    base[2] - support_drop  # Lower (more negative Z)
                ]
                dance_all_legs.append([[lowered[0], lowered[1], lowered[2]]])
        
        dance_scheme = Movements('diagonal_lift')
        dance_speed = [[0, 0, 0]]
        
        # Add body attitude compensation to lean toward support legs
        # For FR_BL lift (support FL+BR): lean left (negative roll) and back (negative pitch)
        # For FL_BR lift (support FR+BL): lean right (positive roll) and forward (positive pitch)
        if pair == "FR_BL":
            lean_roll = -config.BODY_LEAN_ROLL   # Lean left toward FL
            lean_pitch = -config.BODY_LEAN_PITCH  # Lean back toward BR
        else:
            lean_roll = config.BODY_LEAN_ROLL    # Lean right toward FR
            lean_pitch = config.BODY_LEAN_PITCH   # Lean forward toward BL
        
        dance_attitude = [[lean_roll, lean_pitch, 0]]
        
        # Use longer transition time for gradual lift - balance loop runs during this
        dance_scheme.setInterpolationNumber(interval_hold)
        dance_scheme.setTransitionTic(interval_lift)
        dance_scheme.setAllSequence(dance_all_legs, dance_speed, dance_attitude)
        
        self.MovementLib.append(dance_scheme)
        return self.MovementLib
    
    def gradual_diagonal_lift(self, pair="FR_BL", height=None, num_steps=10, 
                               time_per_step=None, time_hold=None):
        """
        Generate a GRADUAL diagonal leg lift with multiple steps.
        
        This creates multiple small lift movements, allowing the balance
        controller to adjust between each step. Much more stable than
        a single large lift.
        
        Args:
            pair: "FR_BL" or "FL_BR" - which diagonal to lift
            height: Total lift height (m), default from config
            num_steps: Number of incremental steps to reach full height
            time_per_step: Time for each step (s), default 0.2s
            time_hold: Time to hold final position (s), default from config
            
        Returns:
            Updated MovementLib
        """
        height = height if height is not None else self.lift_height
        time_per_step = time_per_step if time_per_step is not None else 0.2
        time_hold = time_hold if time_hold is not None else self.hold_time
        
        # Clamp height for safety
        height = self.cap_limit(self.legliftcap, 0, height)
        
        diagonal_config = self.get_diagonal_config(pair)
        lift_legs = diagonal_config["lift"]
        support_legs = diagonal_config["support"]
        
        # Height increment per step
        height_step = height / num_steps
        
        # Calculate body lean (applied gradually)
        if pair == "FR_BL":
            final_lean_roll = -config.BODY_LEAN_ROLL
            final_lean_pitch = -config.BODY_LEAN_PITCH
        else:
            final_lean_roll = config.BODY_LEAN_ROLL
            final_lean_pitch = config.BODY_LEAN_PITCH
        
        lean_roll_step = final_lean_roll / num_steps
        lean_pitch_step = final_lean_pitch / num_steps
        
        interval_step = max(1, int(time_per_step / self.dt))
        interval_hold = max(1, int(time_hold / self.dt))
        
        # Create incremental lift movements
        for step in range(1, num_steps + 1):
            current_height = height_step * step
            current_lean_roll = lean_roll_step * step
            current_lean_pitch = lean_pitch_step * step
            
            dance_all_legs = []
            for leg_idx in range(4):
                base = self.default_stance[leg_idx][0].copy()
                
                if leg_idx in lift_legs:
                    # Gradually lift - current step height
                    forward_shift = 0.02 * (step / num_steps)  # Also gradual
                    lifted = [
                        base[0] + forward_shift,
                        base[1],
                        base[2] + current_height
                    ]
                    dance_all_legs.append([[lifted[0], lifted[1], lifted[2]]])
                else:
                    # Support leg - gradual drop
                    support_drop = 0.015 * (step / num_steps)
                    lowered = [
                        base[0],
                        base[1],
                        base[2] - support_drop
                    ]
                    dance_all_legs.append([[lowered[0], lowered[1], lowered[2]]])
            
            # Name each step for debugging
            step_name = f'gradual_lift_step_{step}'
            if step == num_steps:
                step_name = 'diagonal_lift'  # Final step uses main name for balance detection
            
            dance_scheme = Movements(step_name)
            dance_speed = [[0, 0, 0]]
            dance_attitude = [[current_lean_roll, current_lean_pitch, 0]]
            
            # Use longer hold on final step
            if step == num_steps:
                dance_scheme.setInterpolationNumber(interval_hold)
            else:
                dance_scheme.setInterpolationNumber(interval_step)
            
            dance_scheme.setTransitionTic(interval_step)
            dance_scheme.setAllSequence(dance_all_legs, dance_speed, dance_attitude)
            
            self.MovementLib.append(dance_scheme)
        
        return self.MovementLib
    
    def balance_correction(self, roll_deg, pitch_deg, time_correct=0.02):
        """
        Generate a balance correction movement.
        
        Adjusts body attitude (roll/pitch) to maintain balance while
        in diagonal stance.
        
        Args:
            roll_deg: Roll correction in degrees
            pitch_deg: Pitch correction in degrees
            time_correct: Time for correction (s)
            
        Returns:
            Updated MovementLib
        """
        interval = max(1, int(time_correct / self.dt))
        
        # Clamp for safety
        roll_deg = self.cap_limit(self.rowcap, -self.rowcap, roll_deg)
        pitch_deg = self.cap_limit(self.pitchcap, -self.pitchcap, pitch_deg)
        
        dance_scheme = Movements('balance_correction')
        
        # Keep current leg positions (this will be applied on top of diagonal stance)
        dance_all_legs = self.default_stance
        dance_speed = [[0, 0, 0]]
        dance_attitude = [[roll_deg, pitch_deg, 0], [roll_deg, pitch_deg, 0]]
        
        dance_scheme.setInterpolationNumber(interval)
        dance_scheme.setTransitionTic(interval)
        dance_scheme.setLegsSequence(dance_all_legs)
        dance_scheme.setAttitudeSequence(dance_attitude)
        
        self.MovementLib.append(dance_scheme)
        return self.MovementLib
    
    def return_to_stand(self, time_return=None):
        """
        Generate a return-to-stand movement.
        
        Safely returns all legs to the default standing position.
        
        Args:
            time_return: Transition time (s), default from config
            
        Returns:
            Updated MovementLib
        """
        time_return = time_return if time_return is not None else self.return_time
        interval = max(1, int(time_return / self.dt))
        
        # All legs return to default stance
        dance_all_legs = []
        for leg_idx in range(4):
            base = self.default_stance[leg_idx][0].copy()
            dance_all_legs.append([[base[0], base[1], base[2]],
                                   [base[0], base[1], base[2]]])
        
        dance_scheme = Movements('return_to_stand')
        dance_speed = [[0, 0, 0]]
        dance_attitude = [[0, 0, 0], [0, 0, 0]]
        
        dance_scheme.setInterpolationNumber(interval)
        dance_scheme.setTransitionTic(interval)
        dance_scheme.setAllSequence(dance_all_legs, dance_speed, dance_attitude)
        
        self.MovementLib.append(dance_scheme)
        return self.MovementLib
    
    def generate_full_sequence(self, pair="FR_BL", gradual=True):
        """
        Generate a complete diagonal balance sequence.
        
        Sequence:
        1. CoM shift (if enabled)
        2. Diagonal lift (gradual or single-step)
        3. Hold (built into diagonal_lift)
        4. Return to stand
        
        Args:
            pair: "FR_BL" or "FL_BR"
            gradual: If True, use gradual lift (recommended for stability)
            
        Returns:
            List of Movement objects
        """
        self.clear_movements()
        
        if config.ENABLE_COM_SHIFT:
            self.com_shift(pair=pair)
        
        if gradual:
            # Use gradual lift - much more stable for balancing
            self.gradual_diagonal_lift(
                pair=pair,
                num_steps=config.GRADUAL_LIFT_STEPS,
                time_per_step=config.GRADUAL_LIFT_STEP_TIME
            )
        else:
            # Original single-step lift
            self.diagonal_lift(pair=pair)
        
        self.return_to_stand()
        
        return self.MovementLib
    
    def get_leg_positions_for_validation(self, pair="FR_BL"):
        """
        Get leg position sequences for kinematics validation.
        
        Returns positions as 3x4 matrices suitable for four_legs_inverse_kinematics.
        
        Args:
            pair: "FR_BL" or "FL_BR"
            
        Returns:
            List of 3x4 numpy arrays representing leg positions at key frames
        """
        height = self.cap_limit(self.legliftcap, 0, self.lift_height)
        diagonal_config = self.get_diagonal_config(pair)
        lift_legs = diagonal_config["lift"]
        
        positions = []
        
        # Frame 1: Default stance
        default = config.DEFAULT_STANCE_MATRIX.copy()
        positions.append(default)
        
        # Frame 2: CoM shift
        if config.ENABLE_COM_SHIFT:
            if pair == "FR_BL":
                x_shift = -config.COM_SHIFT_X
                y_shift = config.COM_SHIFT_Y
            else:
                x_shift = config.COM_SHIFT_X
                y_shift = -config.COM_SHIFT_Y
            
            com_shifted = default.copy()
            com_shifted[0, :] -= x_shift
            com_shifted[1, :] -= y_shift
            positions.append(com_shifted)
        
        # Frame 3: Diagonal lifted
        lifted = positions[-1].copy()
        for leg_idx in lift_legs:
            lifted[0, leg_idx] += 0.02  # Forward shift
            lifted[2, leg_idx] += height  # Lift
        
        # Support legs drop slightly
        for leg_idx in diagonal_config["support"]:
            lifted[2, leg_idx] -= 0.01
        
        positions.append(lifted)
        
        # Frame 4: Return to default
        positions.append(default.copy())
        
        return positions


# Convenience function to create a generator with defaults
def create_diagonal_movement_lib(pair=None):
    """
    Create a complete diagonal balance movement library.
    
    Args:
        pair: "FR_BL" or "FL_BR", default from config
        
    Returns:
        List of Movement objects
    """
    pair = pair if pair is not None else config.DIAGONAL_PAIR
    generator = DiagonalMovementGenerator()
    return generator.generate_full_sequence(pair=pair)


if __name__ == "__main__":
    # Test movement generation
    print("Testing Diagonal Movement Generator...")
    
    generator = DiagonalMovementGenerator()
    movements = generator.generate_full_sequence(pair="FR_BL")
    
    print(f"Generated {len(movements)} movements:")
    for i, m in enumerate(movements):
        print(f"  {i+1}. {m.getMovementName()} - {m.getCycleTicks()} ticks")
    
    print("\nLeg positions for validation:")
    positions = generator.get_leg_positions_for_validation(pair="FR_BL")
    for i, pos in enumerate(positions):
        print(f"\nFrame {i+1}:")
        print(f"  X: {pos[0, :]}")
        print(f"  Y: {pos[1, :]}")
        print(f"  Z: {pos[2, :]}")
