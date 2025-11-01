"""
Recovery strategies for GMCS system failures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import copy


# ============================================================================
# Recovery Strategy Base
# ============================================================================

class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""
    
    @abstractmethod
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the error."""
        pass
    
    @abstractmethod
    def recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery and return new context."""
        pass


# ============================================================================
# State Rollback Strategy
# ============================================================================

class StateRollbackStrategy(RecoveryStrategy):
    """
    Rollback system state to last known good checkpoint.
    """
    
    def __init__(self, max_rollbacks: int = 3):
        self.max_rollbacks = max_rollbacks
        self.state_history = []
        self.rollback_count = 0
    
    def save_checkpoint(self, state: Any):
        """Save state checkpoint."""
        # Keep only last N states
        if len(self.state_history) >= 10:
            self.state_history.pop(0)
        self.state_history.append(copy.deepcopy(state))
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Can recover if we have checkpoints and haven't exceeded max rollbacks."""
        return (
            len(self.state_history) > 0 and
            self.rollback_count < self.max_rollbacks
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback to last checkpoint."""
        if not self.state_history:
            raise ValueError("No checkpoints available for rollback")
        
        last_good_state = self.state_history[-1]
        self.rollback_count += 1
        
        print(f"[Recovery] Rolling back to checkpoint ({self.rollback_count}/{self.max_rollbacks})")
        
        return {
            'state': last_good_state,
            'action': 'rollback',
            'rollback_count': self.rollback_count
        }
    
    def reset(self):
        """Reset rollback counter."""
        self.rollback_count = 0


# ============================================================================
# Node Isolation Strategy
# ============================================================================

class NodeIsolationStrategy(RecoveryStrategy):
    """
    Isolate and restart failing nodes while continuing simulation.
    """
    
    def __init__(self):
        self.isolated_nodes = set()
        self.node_failure_counts = {}
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Can recover if error is node-specific."""
        return 'node_id' in context
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate failing node."""
        node_id = context.get('node_id')
        
        if not node_id:
            raise ValueError("Node ID required for isolation")
        
        # Track failures
        self.node_failure_counts[node_id] = self.node_failure_counts.get(node_id, 0) + 1
        
        if self.node_failure_counts[node_id] > 5:
            # Permanently disable after too many failures
            self.isolated_nodes.add(node_id)
            print(f"[Recovery] Node {node_id} permanently isolated after {self.node_failure_counts[node_id]} failures")
            
            return {
                'action': 'isolate_permanent',
                'node_id': node_id,
                'failure_count': self.node_failure_counts[node_id]
            }
        else:
            # Temporary isolation and restart
            print(f"[Recovery] Restarting node {node_id} (failure #{self.node_failure_counts[node_id]})")
            
            return {
                'action': 'restart_node',
                'node_id': node_id,
                'failure_count': self.node_failure_counts[node_id]
            }
    
    def is_isolated(self, node_id: str) -> bool:
        """Check if node is isolated."""
        return node_id in self.isolated_nodes


# ============================================================================
# Parameter Adjustment Strategy
# ============================================================================

class ParameterAdjustmentStrategy(RecoveryStrategy):
    """
    Automatically adjust parameters when instability detected.
    """
    
    def __init__(self):
        self.adjustment_history = []
        self.safety_parameters = {
            'dt': (0.001, 0.1),  # (min, max)
            'coupling_strength': (0.0, 1.0),
            'temperature': (0.1, 10.0)
        }
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Can recover if error is stability-related."""
        error_msg = str(error).lower()
        return any(keyword in error_msg for keyword in [
            'nan', 'inf', 'unstable', 'diverge', 'overflow'
        ])
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters for stability."""
        adjustments = {}
        
        # Reduce time step if present
        if 'dt' in context:
            current_dt = context['dt']
            min_dt, max_dt = self.safety_parameters['dt']
            new_dt = max(current_dt * 0.5, min_dt)
            adjustments['dt'] = new_dt
            print(f"[Recovery] Reducing dt: {current_dt} -> {new_dt}")
        
        # Reduce coupling if present
        if 'coupling_strength' in context:
            current_coupling = context['coupling_strength']
            min_coupling, max_coupling = self.safety_parameters['coupling_strength']
            new_coupling = max(current_coupling * 0.7, min_coupling)
            adjustments['coupling_strength'] = new_coupling
            print(f"[Recovery] Reducing coupling: {current_coupling} -> {new_coupling}")
        
        # Increase temperature for exploration
        if 'temperature' in context:
            current_temp = context['temperature']
            min_temp, max_temp = self.safety_parameters['temperature']
            new_temp = min(current_temp * 1.5, max_temp)
            adjustments['temperature'] = new_temp
            print(f"[Recovery] Increasing temperature: {current_temp} -> {new_temp}")
        
        self.adjustment_history.append({
            'error': str(error),
            'adjustments': adjustments
        })
        
        return {
            'action': 'adjust_parameters',
            'adjustments': adjustments
        }


# ============================================================================
# Recovery Manager
# ============================================================================

class RecoveryManager:
    """
    Manages multiple recovery strategies.
    """
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = [
            ParameterAdjustmentStrategy(),
            NodeIsolationStrategy(),
            StateRollbackStrategy()
        ]
        
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """Add custom recovery strategy."""
        self.strategies.insert(0, strategy)  # Higher priority
    
    def attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt recovery using available strategies.
        
        Args:
            error: The exception that occurred
            context: Context information
            
        Returns:
            Recovery result or None if no strategy worked
        """
        for strategy in self.strategies:
            if strategy.can_recover(error, context):
                try:
                    result = strategy.recover(error, context)
                    self.recovery_stats['total_recoveries'] += 1
                    self.recovery_stats['successful_recoveries'] += 1
                    return result
                except Exception as e:
                    print(f"[RecoveryManager] Strategy {strategy.__class__.__name__} failed: {e}")
                    self.recovery_stats['failed_recoveries'] += 1
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            **self.recovery_stats,
            'strategies_count': len(self.strategies)
        }


# ============================================================================
# Global Recovery Manager
# ============================================================================

_global_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """Get or create global recovery manager."""
    global _global_recovery_manager
    
    if _global_recovery_manager is None:
        _global_recovery_manager = RecoveryManager()
    
    return _global_recovery_manager

