"""
Safety Manager for Diagonal Balance Experiment.

Provides:
- State machine for balance operation
- Abort conditions and emergency stop
- Sustained trigger detection
- Timeout handling
"""

import time
from enum import Enum, auto
from experiments.diagonal_balance import config


class BalanceState(Enum):
    """States for the diagonal balance state machine."""
    IDLE = auto()           # Waiting to start
    PREPARING = auto()      # Checking conditions, sustained trigger
    COM_SHIFTING = auto()   # Shifting center of mass
    LIFTING = auto()        # Raising diagonal legs
    BALANCING = auto()      # Active balance control
    RETURNING = auto()      # Returning to stand
    ABORTED = auto()        # Emergency stop triggered
    COMPLETE = auto()       # Successfully completed


class AbortReason(Enum):
    """Reasons for aborting the balance operation."""
    NONE = auto()
    MAX_TILT_EXCEEDED = auto()
    TIMEOUT = auto()
    USER_ABORT = auto()
    HARDWARE_ERROR = auto()
    UNSTABLE = auto()


class SafetyManager:
    """
    Manages safety conditions and state transitions for diagonal balance.
    """
    
    def __init__(self):
        """Initialize the safety manager."""
        self.state = BalanceState.IDLE
        self.abort_reason = AbortReason.NONE
        
        # Timing
        self.start_time = None
        self.state_start_time = None
        
        # Sustained trigger counter
        self.stable_count = 0
        self.required_stable_count = config.SUSTAINED_TRIGGER_CYCLES
        
        # Thresholds from config
        self.max_tilt = config.MAX_TILT_ABORT
        self.tilt_threshold = config.TILT_THRESHOLD
        self.timeout = config.TIMEOUT_S
        
        # User abort flag
        self.user_abort_requested = False
        
        # State history for debugging
        self.state_history = []
    
    def reset(self):
        """Reset the safety manager to initial state."""
        self.state = BalanceState.IDLE
        self.abort_reason = AbortReason.NONE
        self.start_time = None
        self.state_start_time = None
        self.stable_count = 0
        self.user_abort_requested = False
        self.state_history = []
    
    def start(self):
        """Start the balance operation."""
        self.reset()
        self.start_time = time.time()
        self.transition_to(BalanceState.PREPARING)
    
    def transition_to(self, new_state):
        """
        Transition to a new state.
        
        Args:
            new_state: BalanceState to transition to
        """
        old_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        
        # Record state history
        self.state_history.append({
            'from': old_state,
            'to': new_state,
            'time': time.time()
        })
        
        if config.DEBUG_PRINT:
            print(f"[Safety] State: {old_state.name} -> {new_state.name}")
    
    def request_abort(self, reason=AbortReason.USER_ABORT):
        """
        Request an abort of the balance operation.
        
        Args:
            reason: AbortReason for the abort
        """
        self.user_abort_requested = True
        self.abort_reason = reason
    
    def check_abort_conditions(self, roll_deg, pitch_deg):
        """
        Check if any abort conditions are met.
        
        Args:
            roll_deg: Current roll angle in degrees
            pitch_deg: Current pitch angle in degrees
            
        Returns:
            True if abort should be triggered, False otherwise
        """
        # Check user abort
        if self.user_abort_requested:
            self.abort_reason = AbortReason.USER_ABORT
            return True
        
        # Check max tilt
        if abs(roll_deg) > self.max_tilt or abs(pitch_deg) > self.max_tilt:
            self.abort_reason = AbortReason.MAX_TILT_EXCEEDED
            if config.DEBUG_PRINT:
                print(f"[Safety] ABORT: Tilt exceeded! Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°")
            return True
        
        # Check timeout
        if self.start_time and (time.time() - self.start_time) > self.timeout:
            self.abort_reason = AbortReason.TIMEOUT
            if config.DEBUG_PRINT:
                print(f"[Safety] ABORT: Timeout after {self.timeout}s")
            return True
        
        return False
    
    def check_sustained_trigger(self, roll_deg, pitch_deg):
        """
        Check if the trigger condition is sustained.
        
        Requires N consecutive cycles of angles within threshold before proceeding.
        
        Args:
            roll_deg: Current roll angle in degrees
            pitch_deg: Current pitch angle in degrees
            
        Returns:
            True if sustained trigger condition is met
        """
        # Check if angles are within acceptable range for starting
        if abs(roll_deg) < self.tilt_threshold and abs(pitch_deg) < self.tilt_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        if config.DEBUG_PRINT and self.stable_count > 0:
            if self.stable_count % 10 == 0:  # Print every 10 cycles
                print(f"[Safety] Stable count: {self.stable_count}/{self.required_stable_count}")
        
        return self.stable_count >= self.required_stable_count
    
    def update(self, roll_deg, pitch_deg, movement_complete=False):
        """
        Update the safety state machine.
        
        Args:
            roll_deg: Current roll angle in degrees
            pitch_deg: Current pitch angle in degrees
            movement_complete: True if current movement has completed
            
        Returns:
            Current BalanceState
        """
        # Check abort conditions in any active state
        if self.state not in [BalanceState.IDLE, BalanceState.ABORTED, BalanceState.COMPLETE]:
            if self.check_abort_conditions(roll_deg, pitch_deg):
                self.transition_to(BalanceState.ABORTED)
                return self.state
        
        # State-specific logic
        if self.state == BalanceState.PREPARING:
            if self.check_sustained_trigger(roll_deg, pitch_deg):
                if config.ENABLE_COM_SHIFT:
                    self.transition_to(BalanceState.COM_SHIFTING)
                else:
                    self.transition_to(BalanceState.LIFTING)
        
        elif self.state == BalanceState.COM_SHIFTING:
            if movement_complete:
                self.transition_to(BalanceState.LIFTING)
        
        elif self.state == BalanceState.LIFTING:
            if movement_complete:
                self.transition_to(BalanceState.BALANCING)
        
        elif self.state == BalanceState.BALANCING:
            # Balancing continues until timeout or abort
            # Check if we should return to stand (could add additional conditions)
            pass
        
        elif self.state == BalanceState.RETURNING:
            if movement_complete:
                self.transition_to(BalanceState.COMPLETE)
        
        return self.state
    
    def should_apply_balance_correction(self, roll_deg, pitch_deg):
        """
        Check if balance correction should be applied.
        
        Args:
            roll_deg: Current roll angle in degrees
            pitch_deg: Current pitch angle in degrees
            
        Returns:
            True if correction should be applied
        """
        if self.state != BalanceState.BALANCING:
            return False
        
        return abs(roll_deg) > self.tilt_threshold or abs(pitch_deg) > self.tilt_threshold
    
    def initiate_return(self):
        """Initiate return to stand sequence."""
        if self.state == BalanceState.BALANCING:
            self.transition_to(BalanceState.RETURNING)
    
    def get_elapsed_time(self):
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_state_duration(self):
        """Get duration in current state."""
        if self.state_start_time is None:
            return 0.0
        return time.time() - self.state_start_time
    
    def is_active(self):
        """Check if balance operation is active."""
        return self.state in [
            BalanceState.PREPARING,
            BalanceState.COM_SHIFTING,
            BalanceState.LIFTING,
            BalanceState.BALANCING,
            BalanceState.RETURNING
        ]
    
    def is_complete(self):
        """Check if balance operation has completed."""
        return self.state in [BalanceState.COMPLETE, BalanceState.ABORTED]
    
    def get_status_string(self):
        """Get a human-readable status string."""
        status = f"State: {self.state.name}"
        if self.start_time:
            status += f" | Elapsed: {self.get_elapsed_time():.1f}s"
        if self.abort_reason != AbortReason.NONE:
            status += f" | Abort: {self.abort_reason.name}"
        return status


class EmergencyStop:
    """
    Emergency stop handler.
    
    Provides a callback-based emergency stop mechanism.
    """
    
    def __init__(self, hardware_interface=None):
        """
        Initialize emergency stop.
        
        Args:
            hardware_interface: HardwareInterface instance for servo control
        """
        self.hardware = hardware_interface
        self.triggered = False
        self.callbacks = []
    
    def register_callback(self, callback):
        """Register a callback to be called on emergency stop."""
        self.callbacks.append(callback)
    
    def trigger(self, reason="Unknown"):
        """
        Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        if self.triggered:
            return
        
        self.triggered = True
        print(f"[EMERGENCY STOP] Triggered: {reason}")
        
        # Deactivate servos if hardware available
        if self.hardware:
            try:
                # Try to use the hardware interface's pi object directly
                if hasattr(self.hardware, 'pi') and hasattr(self.hardware, 'pwm_params'):
                    for leg_index in range(4):
                        for axis_index in range(3):
                            pin = self.hardware.pwm_params.pins[axis_index, leg_index]
                            self.hardware.pi.set_PWM_dutycycle(pin, 0)
                    print("[EMERGENCY STOP] Servos deactivated")
            except Exception as e:
                print(f"[EMERGENCY STOP] Failed to deactivate servos: {e}")
        
        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(reason)
            except Exception as e:
                print(f"[EMERGENCY STOP] Callback error: {e}")
    
    def reset(self):
        """Reset emergency stop state."""
        self.triggered = False


if __name__ == "__main__":
    # Test safety manager
    print("Testing Safety Manager...")
    
    safety = SafetyManager()
    safety.start()
    
    print(f"Initial state: {safety.state.name}")
    
    # Simulate sustained trigger
    print("\nSimulating sustained trigger...")
    for i in range(config.SUSTAINED_TRIGGER_CYCLES + 5):
        state = safety.update(roll_deg=0.5, pitch_deg=0.3)
        if state == BalanceState.COM_SHIFTING or state == BalanceState.LIFTING:
            print(f"Triggered at cycle {i+1}!")
            break
    
    print(f"\nFinal state: {safety.state.name}")
    print(f"Status: {safety.get_status_string()}")
    
    # Test abort
    print("\nTesting abort condition...")
    safety.reset()
    safety.start()
    safety.update(roll_deg=0.5, pitch_deg=0.3)
    
    # Simulate excessive tilt
    state = safety.update(roll_deg=20.0, pitch_deg=5.0)
    print(f"After excessive tilt: {state.name}")
    print(f"Abort reason: {safety.abort_reason.name}")
