"""
Comprehensive preset validation tests.

Tests all 15 presets for structure, validity, and basic functionality.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any


PRESET_DIR = Path("frontend/presets")
REQUIRED_PRESET_FIELDS = ["id", "name", "version", "description", "category", "nodes", "connections"]
REQUIRED_NODE_FIELDS = ["id", "type", "name", "position", "config"]
REQUIRED_CONNECTION_FIELDS = ["from", "to"]


def load_preset(preset_file: Path) -> Dict[str, Any]:
    """Load and parse a preset JSON file."""
    with open(preset_file) as f:
        return json.load(f)


def get_all_presets():
    """Get all preset files."""
    if not PRESET_DIR.exists():
        pytest.skip(f"Preset directory not found: {PRESET_DIR}")
    
    presets = list(PRESET_DIR.glob("*.json"))
    if not presets:
        pytest.skip(f"No presets found in {PRESET_DIR}")
    
    return presets


@pytest.mark.parametrize("preset_file", get_all_presets())
class TestPresetStructure:
    """Test preset file structure and validity."""
    
    def test_preset_loads(self, preset_file):
        """Test that preset JSON is valid and loadable."""
        data = load_preset(preset_file)
        assert isinstance(data, dict), f"{preset_file.name}: Must be a JSON object"
    
    def test_required_fields(self, preset_file):
        """Test that all required fields are present."""
        data = load_preset(preset_file)
        
        missing = [f for f in REQUIRED_PRESET_FIELDS if f not in data]
        assert not missing, f"{preset_file.name}: Missing required fields: {missing}"
    
    def test_preset_id_matches_filename(self, preset_file):
        """Test that preset ID matches filename."""
        data = load_preset(preset_file)
        expected_id = preset_file.stem
        actual_id = data.get("id")
        
        assert actual_id == expected_id, \
            f"{preset_file.name}: ID '{actual_id}' doesn't match filename '{expected_id}'"
    
    def test_nodes_structure(self, preset_file):
        """Test that nodes array is properly structured."""
        data = load_preset(preset_file)
        nodes = data.get("nodes", [])
        
        assert isinstance(nodes, list), f"{preset_file.name}: 'nodes' must be an array"
        assert len(nodes) > 0, f"{preset_file.name}: Must have at least one node"
        
        for i, node in enumerate(nodes):
            # Check required fields
            missing = [f for f in REQUIRED_NODE_FIELDS if f not in node]
            assert not missing, \
                f"{preset_file.name}: Node {i} missing fields: {missing}"
            
            # Check node ID uniqueness
            node_ids = [n["id"] for n in nodes]
            assert len(node_ids) == len(set(node_ids)), \
                f"{preset_file.name}: Duplicate node IDs found"
            
            # Check position format
            pos = node.get("position")
            assert isinstance(pos, dict), \
                f"{preset_file.name}: Node {i} position must be object"
            assert "x" in pos and "y" in pos, \
                f"{preset_file.name}: Node {i} position must have x and y"
    
    def test_connections_structure(self, preset_file):
        """Test that connections array is properly structured."""
        data = load_preset(preset_file)
        connections = data.get("connections", [])
        
        assert isinstance(connections, list), \
            f"{preset_file.name}: 'connections' must be an array"
        
        node_ids = {node["id"] for node in data.get("nodes", [])}
        
        for i, conn in enumerate(connections):
            # Check required fields
            missing = [f for f in REQUIRED_CONNECTION_FIELDS if f not in conn]
            assert not missing, \
                f"{preset_file.name}: Connection {i} missing fields: {missing}"
            
            # Extract node IDs from connection strings (format: "node_id.output")
            from_node = conn["from"].split(".")[0]
            to_node = conn["to"].split(".")[0]
            
            # Check that referenced nodes exist
            assert from_node in node_ids, \
                f"{preset_file.name}: Connection {i} references non-existent source node: {from_node}"
            assert to_node in node_ids, \
                f"{preset_file.name}: Connection {i} references non-existent target node: {to_node}"
    
    def test_metadata_fields(self, preset_file):
        """Test metadata fields."""
        data = load_preset(preset_file)
        
        # Check version format
        version = data.get("version")
        assert isinstance(version, str), \
            f"{preset_file.name}: Version must be a string"
        
        # Check category
        valid_categories = ["Creative", "Scientific", "Research", "Production", "Experimental"]
        category = data.get("category")
        assert category in valid_categories, \
            f"{preset_file.name}: Invalid category '{category}'"
        
        # Check tags if present
        if "tags" in data:
            assert isinstance(data["tags"], list), \
                f"{preset_file.name}: Tags must be an array"


class TestPresetSemantics:
    """Test preset semantic correctness."""
    
    def test_no_circular_dependencies(self):
        """Test that presets don't have circular dependencies."""
        for preset_file in get_all_presets():
            data = load_preset(preset_file)
            
            # Build adjacency list
            graph = {}
            for node in data.get("nodes", []):
                graph[node["id"]] = []
            
            for conn in data.get("connections", []):
                from_node = conn["from"].split(".")[0]
                to_node = conn["to"].split(".")[0]
                graph[from_node].append(to_node)
            
            # Check for cycles using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(node):
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
                
                rec_stack.remove(node)
                return False
            
            for node_id in graph:
                if node_id not in visited:
                    assert not has_cycle(node_id), \
                        f"{preset_file.name}: Circular dependency detected involving node {node_id}"
    
    def test_input_nodes_have_no_inputs(self):
        """Test that input nodes don't have incoming connections."""
        for preset_file in get_all_presets():
            data = load_preset(preset_file)
            
            # Find input nodes
            input_nodes = {
                node["id"] for node in data.get("nodes", [])
                if node.get("type") == "input"
            }
            
            # Check connections
            for conn in data.get("connections", []):
                to_node = conn["to"].split(".")[0]
                assert to_node not in input_nodes, \
                    f"{preset_file.name}: Input node {to_node} has incoming connection"
    
    def test_output_nodes_have_no_outputs(self):
        """Test that output nodes don't have outgoing connections."""
        for preset_file in get_all_presets():
            data = load_preset(preset_file)
            
            # Find output nodes
            output_nodes = {
                node["id"] for node in data.get("nodes", [])
                if node.get("type") == "output"
            }
            
            # Check connections
            for conn in data.get("connections", []):
                from_node = conn["from"].split(".")[0]
                assert from_node not in output_nodes, \
                    f"{preset_file.name}: Output node {from_node} has outgoing connection"


class TestSpecificPresets:
    """Test specific preset functionality."""
    
    def test_live_music_preset(self):
        """Test the live_music preset specifically."""
        preset_file = PRESET_DIR / "live_music.json"
        if not preset_file.exists():
            pytest.skip("live_music.json not found")
        
        data = load_preset(preset_file)
        
        # Should have audio input
        node_types = {node["type"] for node in data["nodes"]}
        assert "input" in node_types, "live_music must have input node"
        
        # Should have audio output
        assert "output" in node_types, "live_music must have output node"
        
        # Should have FFT analyzer
        node_names = [node["name"].lower() for node in data["nodes"]]
        has_fft = any("fft" in name or "analyzer" in name for name in node_names)
        assert has_fft, "live_music should have FFT analysis"
    
    def test_world_gen_preset(self):
        """Test the world_gen preset."""
        preset_file = PRESET_DIR / "world_gen.json"
        if not preset_file.exists():
            pytest.skip("world_gen.json not found")
        
        data = load_preset(preset_file)
        
        # Should have reasonable number of nodes for world generation
        assert len(data["nodes"]) >= 5, "world_gen should have multiple processing stages"
    
    def test_all_presets_have_documentation(self):
        """Test that all presets have documentation field."""
        for preset_file in get_all_presets():
            data = load_preset(preset_file)
            
            # Either have documentation field or description
            has_docs = "documentation" in data or "description" in data
            assert has_docs, \
                f"{preset_file.name}: Must have documentation or description"


def test_preset_count():
    """Test that we have the expected number of presets."""
    presets = get_all_presets()
    
    # Should have 15 presets based on WHAT_IS_MISSING.md
    assert len(presets) >= 15, \
        f"Expected at least 15 presets, found {len(presets)}"


def test_all_expected_presets_exist():
    """Test that all expected presets exist."""
    expected_presets = [
        "emergent_logic",
        "live_music",
        "music_analysis",
        "world_gen",
        "nas_discovery",
        "rl_boost",
        "pixel_art",
        "molecular_design",
        "quantum_opt",
        "solar_opt",
        "photonic_sim",
        "neuromapping",
        "neural_sim",
        "game_code",
        "live_video",
    ]
    
    existing_presets = [p.stem for p in get_all_presets()]
    
    for expected in expected_presets:
        assert expected in existing_presets, \
            f"Expected preset '{expected}.json' not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

