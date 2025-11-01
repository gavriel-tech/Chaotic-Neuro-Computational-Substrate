"""
Comprehensive tests for newly implemented nodes.

Tests all Phase 1-3 nodes: Music, ML, and Crypto nodes.
"""

import pytest
import numpy as np


# ============================================================================
# Music Node Tests
# ============================================================================

class TestMusicNodes:
    """Test music processing nodes."""
    
    def test_chord_detector(self):
        """Test chord detection."""
        from src.processor.music_nodes import ChordDetector
        
        detector = ChordDetector({'method': 'chromagram'})
        
        # Create mock spectrum (simulating C major chord)
        spectrum = np.zeros(1025)
        spectrum[100] = 1.0  # C
        spectrum[125] = 0.8  # E
        spectrum[150] = 0.8  # G
        
        result = detector.process(spectrum)
        
        assert 'chord' in result
        assert 'confidence' in result
        assert isinstance(result['confidence'], float)
    
    def test_beat_tracker(self):
        """Test beat tracking."""
        from src.processor.music_nodes import BeatTracker
        
        tracker = BeatTracker({'sample_rate': 48000})
        
        # Simulate several frames
        for i in range(50):
            spectrum = np.random.randn(1025) * 0.1
            result = tracker.process(spectrum, time=i * 0.01)
            
            assert 'bpm' in result
            assert 'beat_detected' in result
            assert isinstance(result['bpm'], float)
    
    def test_harmony_analyzer(self):
        """Test harmony analysis."""
        from src.processor.music_nodes import HarmonyAnalyzer
        
        analyzer = HarmonyAnalyzer({'window_size': 100})
        
        # Create chroma features (C major)
        chroma = np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        
        result = analyzer.process(chroma)
        
        assert 'key' in result
        assert 'mode' in result
        assert 'stability' in result
    
    def test_rhythm_generator(self):
        """Test rhythm generation."""
        from src.processor.music_nodes import RhythmGenerator
        
        generator = RhythmGenerator({'method': 'euclidean'})
        
        result = generator.process(steps=16, pulses=4)
        
        assert 'pattern' in result
        assert len(result['pattern']) == 16
        assert sum(result['pattern']) == 4  # 4 pulses


# ============================================================================
# ML Model Tests
# ============================================================================

class TestMLModels:
    """Test ML models."""
    
    def test_genre_classifier(self):
        """Test genre classifier."""
        from src.ml.concrete_models import GenreClassifier
        
        config = {
            'class_names': ['rock', 'pop', 'classical']
        }
        
        try:
            model = GenreClassifier('test_node', config)
            
            # Mock spectrogram
            spec = np.random.randn(1, 128, 128)
            result = model.process(spec)
            
            assert 'genre' in result
            assert 'probabilities' in result
            assert 'confidence' in result
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_value_function(self):
        """Test value function."""
        from src.ml.concrete_models import ValueFunction
        
        config = {
            'input_dim': 64,
            'hidden_dims': [128, 64]
        }
        
        try:
            model = ValueFunction('test_node', config)
            
            # Mock state
            state = np.random.randn(64)
            result = model.process(state)
            
            assert 'value' in result
            assert isinstance(result['value'], float)
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_logic_gate_detector(self):
        """Test logic gate detector."""
        from src.ml.concrete_models import LogicGateDetector
        
        config = {
            'seq_length': 200,
            'input_features': 3
        }
        
        try:
            model = LogicGateDetector('test_node', config)
            
            # Mock time series
            time_series = np.random.randn(200, 3)
            result = model.process(time_series)
            
            assert 'gate_type' in result
            assert 'confidence' in result
            assert result['gate_type'] in ['AND', 'OR', 'XOR', 'NAND']
        except ImportError:
            pytest.skip("PyTorch not available")


# ============================================================================
# Crypto Node Tests
# ============================================================================

class TestCryptoNodes:
    """Test cryptography nodes."""
    
    def test_stream_cipher(self):
        """Test chaos stream cipher."""
        from src.processor.crypto_nodes import ChaosStreamCipher
        
        cipher = ChaosStreamCipher({})
        
        # Set key
        key = b'test_key_12345'
        cipher.set_key(key)
        
        # Encrypt
        plaintext = b'Hello, World!'
        result = cipher.process(plaintext, mode='encrypt')
        ciphertext = result['output']
        
        # Decrypt
        cipher.set_key(key)  # Reset cipher
        result = cipher.process(ciphertext, mode='decrypt')
        decrypted = result['output']
        
        assert decrypted == plaintext
    
    def test_hash_function(self):
        """Test chaos hash function."""
        from src.processor.crypto_nodes import HashFunction
        
        hasher = HashFunction({'output_bits': 256})
        
        data = b'test data'
        result = hasher.process(data)
        
        assert 'hash' in result
        assert 'hex_digest' in result
        assert len(result['hash']) == 32  # 256 bits = 32 bytes
    
    def test_key_derivation(self):
        """Test key derivation."""
        from src.processor.crypto_nodes import KeyDerivation
        
        kdf = KeyDerivation({'iterations': 1000, 'key_length': 32})
        
        password = b'my_password'
        result = kdf.process(password)
        
        assert 'key' in result
        assert 'salt' in result
        assert len(result['key']) == 32
    
    def test_random_number_generator(self):
        """Test crypto RNG."""
        from src.processor.crypto_nodes import RandomNumberGenerator
        
        rng = RandomNumberGenerator({'oscillator_count': 16})
        rng.seed(b'test_seed')
        
        result = rng.process(num_bytes=32)
        
        assert 'random_bytes' in result
        assert 'entropy_estimate' in result
        assert len(result['random_bytes']) == 32
    
    def test_crypto_analyzer(self):
        """Test crypto analyzer."""
        from src.processor.crypto_nodes import CryptoAnalyzer
        
        analyzer = CryptoAnalyzer({
            'tests': ['frequency', 'runs', 'entropy']
        })
        
        # Generate some data
        data = np.random.randint(0, 256, size=1000, dtype=np.uint8).tobytes()
        
        result = analyzer.process(data)
        
        assert 'overall' in result
        assert 'frequency_test' in result or 'runs_test' in result


# ============================================================================
# Infrastructure Tests
# ============================================================================

class TestInfrastructure:
    """Test production infrastructure."""
    
    def test_error_handler(self):
        """Test error handler."""
        from src.recovery.error_handler import ErrorHandler, RetryConfig
        
        handler = ErrorHandler()
        
        # Test retry
        call_count = [0]
        
        @handler.with_retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        def failing_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary error")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count[0] == 2
    
    def test_checkpoint_manager(self):
        """Test checkpoint manager."""
        from src.checkpoint.checkpoint_manager import CheckpointManager, CheckpointConfig
        
        config = CheckpointConfig(checkpoint_dir='test_checkpoints', compress=False)
        manager = CheckpointManager(config)
        
        # Save checkpoint
        test_state = {'value': 42, 'name': 'test'}
        filepath = manager.save_checkpoint(test_state, name='test_checkpoint')
        
        # Load checkpoint
        loaded = manager.load_checkpoint('test_checkpoint')
        
        assert loaded['state'] == test_state
        
        # Cleanup
        import shutil
        shutil.rmtree('test_checkpoints', ignore_errors=True)
    
    def test_config_manager(self):
        """Test configuration manager."""
        try:
            from src.config.config_manager import ConfigManager
            
            manager = ConfigManager()
            
            # Test get/set
            original_port = manager.get('server.port', 8000)
            manager.set('server.port', 9000)
            assert manager.get('server.port') == 9000
            
            # Reset
            manager.set('server.port', original_port)
        except ImportError:
            pytest.skip("Pydantic not available")
    
    def test_logging_setup(self):
        """Test logging configuration."""
        from src.logging.structured_logger import configure_logging, get_logger
        
        configure_logging(log_dir='test_logs', level='DEBUG', console_output=False)
        
        logger = get_logger('test')
        logger.info("Test message")
        
        # Cleanup
        import shutil
        shutil.rmtree('test_logs', ignore_errors=True)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with presets."""
    
    def test_music_pipeline(self):
        """Test complete music analysis pipeline."""
        from src.processor.music_nodes import ChordDetector, BeatTracker
        
        detector = ChordDetector({})
        tracker = BeatTracker({})
        
        # Simulate audio processing pipeline
        for i in range(10):
            spectrum = np.random.randn(1025)
            
            chord_result = detector.process(spectrum)
            beat_result = tracker.process(spectrum, time=i * 0.02)
            
            assert chord_result is not None
            assert beat_result is not None
    
    def test_crypto_pipeline(self):
        """Test complete cryptography pipeline."""
        from src.processor.crypto_nodes import (
            RandomNumberGenerator,
            HashFunction,
            ChaosStreamCipher
        )
        
        # Generate key
        rng = RandomNumberGenerator({})
        rng.seed(b'master_seed')
        key_result = rng.process(num_bytes=32)
        key = key_result['random_bytes']
        
        # Hash data
        hasher = HashFunction({})
        data = b'Important data'
        hash_result = hasher.process(data)
        
        # Encrypt data
        cipher = ChaosStreamCipher({})
        cipher.set_key(key)
        encrypted = cipher.process(data)
        
        assert len(hash_result['hash']) > 0
        assert encrypted['output'] != data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

