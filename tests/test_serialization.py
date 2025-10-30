"""
Tests for binary serialization.
"""

import pytest
import jax
import numpy as np
from src.core.state import initialize_system_state, N_MAX
from src.core.simulation import add_node_to_state
from src.api.serializers import (
    serialize_for_frontend,
    deserialize_packet_info,
    compute_packet_size,
    get_adaptive_downsample_resolution,
    estimate_bandwidth,
)


def test_serialize_for_frontend():
    """Test basic serialization."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    packet = serialize_for_frontend(state, down_w=128, down_h=128)
    
    assert isinstance(packet, bytes)
    assert len(packet) > 0


def test_serialize_packet_size():
    """Test that packet size matches expected."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    down_w, down_h = 128, 128
    packet = serialize_for_frontend(state, down_w=down_w, down_h=down_h)
    
    expected_size = compute_packet_size(down_w, down_h)
    assert len(packet) == expected_size


def test_deserialize_packet_info():
    """Test packet header deserialization."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    down_w, down_h = 64, 64
    packet = serialize_for_frontend(state, down_w=down_w, down_h=down_h)
    
    info = deserialize_packet_info(packet)
    
    assert info['down_w'] == down_w
    assert info['down_h'] == down_h
    assert info['n_osc'] == N_MAX
    assert info['n_pos'] == N_MAX
    assert info['valid'] is True


def test_serialization_with_active_nodes():
    """Test serialization with active nodes."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add some nodes
    state, _ = add_node_to_state(state, (100.0, 100.0))
    state, _ = add_node_to_state(state, (150.0, 150.0))
    
    packet = serialize_for_frontend(state, down_w=128, down_h=128)
    
    assert len(packet) > 0
    info = deserialize_packet_info(packet)
    assert info['valid'] is True


def test_compute_packet_size():
    """Test packet size computation."""
    size_128 = compute_packet_size(down_w=128, down_h=128)
    size_64 = compute_packet_size(down_w=64, down_h=64)
    
    # Smaller resolution should give smaller packet
    assert size_64 < size_128
    
    # Should be positive
    assert size_128 > 0


def test_get_adaptive_downsample_resolution():
    """Test adaptive resolution selection."""
    # Single client - full resolution
    res_1 = get_adaptive_downsample_resolution(1)
    assert res_1 == (256, 256)
    
    # Few clients - half resolution
    res_3 = get_adaptive_downsample_resolution(3)
    assert res_3 == (128, 128)
    
    # Many clients - quarter resolution
    res_10 = get_adaptive_downsample_resolution(10)
    assert res_10 == (64, 64)
    
    # Very many clients - minimal resolution
    res_20 = get_adaptive_downsample_resolution(20)
    assert res_20 == (32, 32)


def test_estimate_bandwidth():
    """Test bandwidth estimation."""
    estimate = estimate_bandwidth(fps=30, down_w=128, down_h=128)
    
    assert 'packet_size_bytes' in estimate
    assert 'bandwidth_mbps' in estimate
    assert estimate['fps'] == 30
    assert estimate['bandwidth_mbps'] > 0


def test_serialization_different_resolutions():
    """Test serialization at different resolutions."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    resolutions = [(256, 256), (128, 128), (64, 64), (32, 32)]
    
    for down_w, down_h in resolutions:
        packet = serialize_for_frontend(state, down_w=down_w, down_h=down_h)
        info = deserialize_packet_info(packet)
        
        assert info['down_w'] == down_w
        assert info['down_h'] == down_h
        assert info['valid'] is True


def test_packet_info_invalid_packet():
    """Test that invalid packets are detected."""
    # Create invalid packet (too short)
    invalid_packet = b'\x00' * 10
    
    with pytest.raises(ValueError):
        deserialize_packet_info(invalid_packet)


def test_bandwidth_scales_with_resolution():
    """Test that bandwidth scales with resolution."""
    bw_low = estimate_bandwidth(fps=30, down_w=64, down_h=64)
    bw_high = estimate_bandwidth(fps=30, down_w=256, down_h=256)
    
    # Higher resolution should require more bandwidth
    assert bw_high['bandwidth_mbps'] > bw_low['bandwidth_mbps']


def test_bandwidth_scales_with_fps():
    """Test that bandwidth scales with FPS."""
    bw_30 = estimate_bandwidth(fps=30, down_w=128, down_h=128)
    bw_60 = estimate_bandwidth(fps=60, down_w=128, down_h=128)
    
    # Higher FPS should require more bandwidth
    assert bw_60['bandwidth_mbps'] > bw_30['bandwidth_mbps']
    # Should be approximately 2x
    assert abs(bw_60['bandwidth_mbps'] / bw_30['bandwidth_mbps'] - 2.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

