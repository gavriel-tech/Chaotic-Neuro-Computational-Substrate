"""
Universal Modulation Matrix System.

Implements bidirectional modulation routing between:
- GMCS algorithms
- THRML p-bit outputs
- External ML models
- Chua oscillators
- Wave field
- Audio inputs

The modulation matrix allows any component to modulate any other component,
creating a universal, programmable signal routing system.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from enum import IntEnum


class ModulationSource(IntEnum):
    """Source types for modulation."""
    OSCILLATOR_X = 0  # Chua x state
    OSCILLATOR_Y = 1  # Chua y state
    OSCILLATOR_Z = 2  # Chua z state
    GMCS_OUTPUT = 3   # GMCS continuous output
    THRML_PBIT = 4    # THRML p-bit state
    WAVE_FIELD = 5    # Wave field sample
    AUDIO_PITCH = 6   # Audio pitch
    AUDIO_RMS = 7     # Audio RMS
    EXTERNAL_MODEL = 8  # External ML model output
    CONSTANT = 9      # Constant value


class ModulationTarget(IntEnum):
    """Target types for modulation."""
    GMCS_PARAM_0 = 0   # A_max
    GMCS_PARAM_1 = 1   # R_comp
    GMCS_PARAM_2 = 2   # T_comp
    GMCS_PARAM_3 = 3   # R_exp
    GMCS_PARAM_4 = 4   # T_exp
    GMCS_PARAM_5 = 5   # Phi
    GMCS_PARAM_6 = 6   # omega
    GMCS_PARAM_7 = 7   # gamma
    GMCS_PARAM_8 = 8   # beta
    GMCS_PARAM_9 = 9   # f0
    GMCS_PARAM_10 = 10  # Q
    GMCS_PARAM_11 = 11  # levels
    GMCS_PARAM_12 = 12  # rate_limit
    GMCS_PARAM_13 = 13  # n2
    GMCS_PARAM_14 = 14  # V
    GMCS_PARAM_15 = 15  # V_pi
    THRML_BIAS = 16    # THRML node bias
    THRML_TEMPERATURE = 17  # THRML sampling temperature
    OSCILLATOR_FORCE = 18  # Direct oscillator driving force
    WAVE_SOURCE_STRENGTH = 19  # Wave field source strength
    WAVE_SPEED = 20    # Wave propagation speed


@dataclass
class ModulationRoute:
    """
    A single modulation routing.
    
    Defines how a source modulates a target with a specific strength and mode.
    """
    source_type: ModulationSource
    source_node_id: int  # Which node/component (or -1 for global)
    target_type: ModulationTarget
    target_node_id: int  # Which node/component (or -1 for global)
    strength: float  # Modulation strength/gain
    mode: str  # 'add', 'multiply', 'replace', 'conditional'
    condition_node_id: int = -1  # For conditional modulation
    condition_value: float = 0.0  # Condition threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_type': int(self.source_type),
            'source_node_id': self.source_node_id,
            'target_type': int(self.target_type),
            'target_node_id': self.target_node_id,
            'strength': float(self.strength),
            'mode': self.mode,
            'condition_node_id': self.condition_node_id,
            'condition_value': float(self.condition_value)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModulationRoute':
        """Create from dictionary."""
        return cls(
            source_type=ModulationSource(data['source_type']),
            source_node_id=data['source_node_id'],
            target_type=ModulationTarget(data['target_type']),
            target_node_id=data['target_node_id'],
            strength=data['strength'],
            mode=data['mode'],
            condition_node_id=data.get('condition_node_id', -1),
            condition_value=data.get('condition_value', 0.0)
        )


class ModulationMatrix:
    """
    Universal modulation matrix for bidirectional signal routing.
    
    Manages a collection of modulation routes and applies them to the system state.
    """
    
    def __init__(self, n_max: int = 1024):
        """
        Initialize modulation matrix.
        
        Args:
            n_max: Maximum number of nodes
        """
        self.n_max = n_max
        self.routes: List[ModulationRoute] = []
        self.external_model_outputs: Dict[int, jnp.ndarray] = {}
        
    def add_route(self, route: ModulationRoute) -> int:
        """
        Add a modulation route.
        
        Args:
            route: ModulationRoute to add
            
        Returns:
            Route ID (index in routes list)
        """
        self.routes.append(route)
        return len(self.routes) - 1
    
    def remove_route(self, route_id: int) -> bool:
        """
        Remove a modulation route.
        
        Args:
            route_id: Route ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if 0 <= route_id < len(self.routes):
            self.routes.pop(route_id)
            return True
        return False
    
    def update_route(self, route_id: int, route: ModulationRoute) -> bool:
        """
        Update an existing route.
        
        Args:
            route_id: Route ID to update
            route: New ModulationRoute
            
        Returns:
            True if updated, False if not found
        """
        if 0 <= route_id < len(self.routes):
            self.routes[route_id] = route
            return True
        return False
    
    def set_external_model_output(self, model_id: int, output: jnp.ndarray):
        """
        Set output from an external ML model.
        
        Args:
            model_id: External model ID
            output: Model output array
        """
        self.external_model_outputs[model_id] = output
    
    def extract_source_value(
        self,
        route: ModulationRoute,
        state: Any,  # SystemState
        audio_pitch: float,
        audio_rms: float
    ) -> float:
        """
        Extract source value from system state.
        
        Args:
            route: Modulation route
            state: SystemState
            audio_pitch: Current audio pitch
            audio_rms: Current audio RMS
            
        Returns:
            Source value
        """
        source_type = route.source_type
        node_id = route.source_node_id
        
        if source_type == ModulationSource.OSCILLATOR_X:
            return float(state.oscillator_state[node_id, 0])
        elif source_type == ModulationSource.OSCILLATOR_Y:
            return float(state.oscillator_state[node_id, 1])
        elif source_type == ModulationSource.OSCILLATOR_Z:
            return float(state.oscillator_state[node_id, 2])
        elif source_type == ModulationSource.GMCS_OUTPUT:
            # Would need to store GMCS outputs in state
            return 0.0  # Placeholder
        elif source_type == ModulationSource.THRML_PBIT:
            # Would need THRML state
            return 0.0  # Placeholder
        elif source_type == ModulationSource.WAVE_FIELD:
            pos = state.node_positions[node_id]
            x, y = int(pos[0]), int(pos[1])
            return float(state.field_p[x, y])
        elif source_type == ModulationSource.AUDIO_PITCH:
            return audio_pitch
        elif source_type == ModulationSource.AUDIO_RMS:
            return audio_rms
        elif source_type == ModulationSource.EXTERNAL_MODEL:
            if node_id in self.external_model_outputs:
                return float(self.external_model_outputs[node_id])
            return 0.0
        elif source_type == ModulationSource.CONSTANT:
            return route.strength  # Use strength as constant value
        else:
            return 0.0
    
    def apply_modulation(
        self,
        state: Any,  # SystemState
        audio_pitch: float = 440.0,
        audio_rms: float = 0.0
    ) -> Any:  # SystemState
        """
        Apply all modulation routes to system state.
        
        Args:
            state: Current SystemState
            audio_pitch: Current audio pitch
            audio_rms: Current audio RMS
            
        Returns:
            Modified SystemState
        """
        # Create mutable copies of parameters
        gmcs_params = {
            0: state.gmcs_A_max.copy(),
            1: state.gmcs_R_comp.copy(),
            2: state.gmcs_T_comp.copy(),
            3: state.gmcs_R_exp.copy(),
            4: state.gmcs_T_exp.copy(),
            5: state.gmcs_Phi.copy(),
            6: state.gmcs_omega.copy(),
            7: state.gmcs_gamma.copy(),
            8: state.gmcs_beta.copy(),
            9: state.gmcs_f0.copy(),
            10: state.gmcs_Q.copy(),
            11: state.gmcs_levels.copy(),
            12: state.gmcs_rate_limit.copy(),
            13: state.gmcs_n2.copy(),
            14: state.gmcs_V.copy(),
            15: state.gmcs_V_pi.copy(),
        }
        
        k_strengths = state.k_strengths.copy()
        c_val = state.c_val.copy()
        
        # Apply each route
        for route in self.routes:
            # Extract source value
            source_value = self.extract_source_value(route, state, audio_pitch, audio_rms)
            
            # Check condition if conditional
            if route.mode == 'conditional':
                if route.condition_node_id >= 0:
                    condition_value = self.extract_source_value(
                        ModulationRoute(
                            ModulationSource.OSCILLATOR_X,
                            route.condition_node_id,
                            route.target_type,
                            route.target_node_id,
                            1.0,
                            'add'
                        ),
                        state,
                        audio_pitch,
                        audio_rms
                    )
                    if condition_value < route.condition_value:
                        continue  # Skip this route
            
            # Apply modulation based on target
            target_type = route.target_type
            target_node = route.target_node_id
            strength = route.strength
            mode = route.mode
            
            # GMCS parameters
            if target_type.value <= 15:  # GMCS_PARAM_0 through GMCS_PARAM_15
                param_id = target_type.value
                if target_node == -1:  # Global (all nodes)
                    if mode == 'add':
                        gmcs_params[param_id] = gmcs_params[param_id] + source_value * strength
                    elif mode == 'multiply':
                        gmcs_params[param_id] = gmcs_params[param_id] * (1.0 + source_value * strength)
                    elif mode == 'replace':
                        gmcs_params[param_id] = jnp.full_like(gmcs_params[param_id], source_value * strength)
                else:  # Specific node
                    if mode == 'add':
                        gmcs_params[param_id] = gmcs_params[param_id].at[target_node].add(source_value * strength)
                    elif mode == 'multiply':
                        gmcs_params[param_id] = gmcs_params[param_id].at[target_node].multiply(1.0 + source_value * strength)
                    elif mode == 'replace':
                        gmcs_params[param_id] = gmcs_params[param_id].at[target_node].set(source_value * strength)
            
            # Wave source strength
            elif target_type == ModulationTarget.WAVE_SOURCE_STRENGTH:
                if target_node == -1:
                    if mode == 'add':
                        k_strengths = k_strengths + source_value * strength
                    elif mode == 'multiply':
                        k_strengths = k_strengths * (1.0 + source_value * strength)
                else:
                    if mode == 'add':
                        k_strengths = k_strengths.at[target_node].add(source_value * strength)
                    elif mode == 'multiply':
                        k_strengths = k_strengths.at[target_node].multiply(1.0 + source_value * strength)
            
            # Wave speed
            elif target_type == ModulationTarget.WAVE_SPEED:
                if mode == 'add':
                    c_val = c_val + source_value * strength
                elif mode == 'multiply':
                    c_val = c_val * (1.0 + source_value * strength)
                elif mode == 'replace':
                    c_val = jnp.array([source_value * strength])
        
        # Update state with modulated parameters
        new_state = state._replace(
            gmcs_A_max=gmcs_params[0],
            gmcs_R_comp=gmcs_params[1],
            gmcs_T_comp=gmcs_params[2],
            gmcs_R_exp=gmcs_params[3],
            gmcs_T_exp=gmcs_params[4],
            gmcs_Phi=gmcs_params[5],
            gmcs_omega=gmcs_params[6],
            gmcs_gamma=gmcs_params[7],
            gmcs_beta=gmcs_params[8],
            gmcs_f0=gmcs_params[9],
            gmcs_Q=gmcs_params[10],
            gmcs_levels=gmcs_params[11],
            gmcs_rate_limit=gmcs_params[12],
            gmcs_n2=gmcs_params[13],
            gmcs_V=gmcs_params[14],
            gmcs_V_pi=gmcs_params[15],
            k_strengths=k_strengths,
            c_val=c_val
        )
        
        return new_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'n_max': self.n_max,
            'routes': [route.to_dict() for route in self.routes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModulationMatrix':
        """Deserialize from dictionary."""
        matrix = cls(n_max=data['n_max'])
        matrix.routes = [ModulationRoute.from_dict(r) for r in data['routes']]
        return matrix


def create_audio_reactive_preset(n_nodes: int) -> ModulationMatrix:
    """
    Create a preset modulation matrix for audio-reactive behavior.
    
    Args:
        n_nodes: Number of active nodes
        
    Returns:
        ModulationMatrix with audio routing
    """
    matrix = ModulationMatrix()
    
    # Audio RMS → Wave source strength (all nodes)
    matrix.add_route(ModulationRoute(
        source_type=ModulationSource.AUDIO_RMS,
        source_node_id=-1,
        target_type=ModulationTarget.WAVE_SOURCE_STRENGTH,
        target_node_id=-1,
        strength=2.0,
        mode='multiply'
    ))
    
    # Audio pitch → GMCS omega (all nodes)
    matrix.add_route(ModulationRoute(
        source_type=ModulationSource.AUDIO_PITCH,
        source_node_id=-1,
        target_type=ModulationTarget.GMCS_PARAM_6,  # omega
        target_node_id=-1,
        strength=0.01,
        mode='multiply'
    ))
    
    # Audio pitch → Wave speed
    matrix.add_route(ModulationRoute(
        source_type=ModulationSource.AUDIO_PITCH,
        source_node_id=-1,
        target_type=ModulationTarget.WAVE_SPEED,
        target_node_id=-1,
        strength=0.001,
        mode='add'
    ))
    
    return matrix


def create_feedback_preset(n_nodes: int) -> ModulationMatrix:
    """
    Create a preset with oscillator self-feedback.
    
    Args:
        n_nodes: Number of active nodes
        
    Returns:
        ModulationMatrix with feedback routing
    """
    matrix = ModulationMatrix()
    
    # Each oscillator's x state → its own GMCS limiter threshold
    for i in range(n_nodes):
        matrix.add_route(ModulationRoute(
            source_type=ModulationSource.OSCILLATOR_X,
            source_node_id=i,
            target_type=ModulationTarget.GMCS_PARAM_0,  # A_max
            target_node_id=i,
            strength=0.1,
            mode='add'
        ))
    
    return matrix

