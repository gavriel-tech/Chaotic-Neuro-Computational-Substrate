"""
Tests for API routes and models.
"""

import pytest
from pydantic import ValidationError
from src.api.routes import (
    AddNodeRequest,
    UpdateNodeRequest,
    merge_config_with_defaults,
    pad_chain_to_max_length,
    validate_parameter_ranges,
    DEFAULT_GMCS_PARAMS,
    MAX_CHAIN_LEN,
)


def test_add_node_request_valid():
    """Test valid AddNodeRequest."""
    request = AddNodeRequest(
        position=[100.0, 100.0],
        config={'A_max': 1.0},
        chain=[1, 2],
        initial_perturbation=0.5
    )
    
    assert request.position == [100.0, 100.0]
    assert request.config == {'A_max': 1.0}
    assert request.chain == [1, 2]


def test_add_node_request_out_of_bounds():
    """Test that out of bounds position is rejected."""
    with pytest.raises(ValidationError):
        AddNodeRequest(position=[1000.0, 100.0])  # x too large


def test_update_node_request_valid():
    """Test valid UpdateNodeRequest."""
    request = UpdateNodeRequest(
        node_ids=[0, 1],
        config_updates={'A_max': 0.5},
        chain_update=[1]
    )
    
    assert request.node_ids == [0, 1]
    assert request.config_updates == {'A_max': 0.5}


def test_update_node_request_invalid_node_id():
    """Test that invalid node ID is rejected."""
    with pytest.raises(ValidationError):
        UpdateNodeRequest(node_ids=[10000])  # Too large


def test_merge_config_with_defaults():
    """Test merging user config with defaults."""
    user_config = {'A_max': 1.5, 'Phi': 0.3}
    merged = merge_config_with_defaults(user_config)
    
    # User values should override
    assert merged['A_max'] == 1.5
    assert merged['Phi'] == 0.3
    
    # Defaults should be present
    assert 'R_comp' in merged
    assert merged['R_comp'] == DEFAULT_GMCS_PARAMS['R_comp']


def test_pad_chain_to_max_length():
    """Test padding algorithm chain."""
    chain = [1, 2, 3]
    padded = pad_chain_to_max_length(chain)
    
    assert len(padded) == MAX_CHAIN_LEN
    assert padded[0] == 1
    assert padded[1] == 2
    assert padded[2] == 3
    assert padded[3] == 0  # NOPs


def test_validate_parameter_ranges():
    """Test parameter validation and clamping."""
    config = {
        'A_max': 10.0,  # Too large, should be clamped to 5.0
        'Phi': -0.5,    # Negative, should be clamped to 0.0
        'R_comp': 3.0,  # Valid, should stay 3.0
    }
    
    validated = validate_parameter_ranges(config)
    
    assert validated['A_max'] == 5.0
    assert validated['Phi'] == 0.0
    assert validated['R_comp'] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

