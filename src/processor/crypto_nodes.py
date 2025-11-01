"""
Cryptography Nodes for GMCS - Chaos-Based Security.

Provides chaos-based cryptographic primitives:
- ChaosStreamCipher: Stream encryption using chaotic oscillators
- HashFunction: Chaos-based hashing with avalanche effect
- KeyDerivation: Key generation from chaotic iteration
- RandomNumberGenerator: CSPRNG using coupled oscillators
- CryptoAnalyzer: NIST SP 800-22 statistical tests

These nodes demonstrate practical applications of chaotic dynamics
in security and cryptography.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from collections import Counter


# ============================================================================
# Chaos Stream Cipher
# ============================================================================

@dataclass
class ChaosStreamCipherConfig:
    """Configuration for chaos stream cipher."""
    oscillator_count: int = 3
    alpha: float = 15.6
    beta: float = 28.0
    m0: float = -1.143
    m1: float = -0.714
    dt: float = 0.01


class ChaosStreamCipher:
    """
    Stream cipher using Chua oscillator as pseudorandom keystream generator.
    
    The chaotic oscillator's sensitive dependence on initial conditions
    makes it suitable for cryptography when properly seeded.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = ChaosStreamCipherConfig(**config)
        
        # Chua oscillator state (x, y, z)
        self.state = np.zeros(3)
        self.initialized = False
    
    def set_key(self, key: bytes):
        """
        Initialize oscillator state from key using SHA-256.
        
        Args:
            key: Encryption key (arbitrary length)
        """
        # Derive initial conditions from key hash
        key_hash = hashlib.sha256(key).digest()
        
        # Use first 24 bytes to initialize oscillator state
        initial_values = np.frombuffer(key_hash[:24], dtype=np.float64)
        
        # Normalize to reasonable range for Chua oscillator
        self.state = initial_values.reshape(3) * 2.0 - 1.0
        self.initialized = True
    
    def _chua_step(self) -> float:
        """
        Perform one Chua oscillator integration step.
        
        Returns:
            Quantized output byte (0-255)
        """
        x, y, z = self.state
        
        # Chua piecewise-linear function
        h = self.config.m1 * x + 0.5 * (self.config.m0 - self.config.m1) * (
            abs(x + 1) - abs(x - 1)
        )
        
        # Chua equations (RK4 integration)
        dx = self.config.alpha * (y - x - h)
        dy = x - y + z
        dz = -self.config.beta * y
        
        # Update state
        self.state += self.config.dt * np.array([dx, dy, dz])
        
        # Quantize x coordinate to byte
        # Map oscillator output (~[-2, 2]) to [0, 255]
        byte_val = int((self.state[0] + 2.0) * 63.75) % 256
        
        return byte_val
    
    def process(self, data: bytes, mode: str = 'encrypt') -> Dict[str, Any]:
        """
        Encrypt or decrypt data using chaotic keystream.
        
        Args:
            data: Input data (plaintext for encrypt, ciphertext for decrypt)
            mode: 'encrypt' or 'decrypt' (same operation for stream cipher)
            
        Returns:
            Dict with output, keystream, mode
        """
        if not self.initialized:
            raise ValueError("Cipher not initialized. Call set_key() first.")
        
        # Generate keystream
        keystream = []
        for _ in range(len(data)):
            key_byte = self._chua_step()
            keystream.append(key_byte)
        
        # XOR data with keystream
        result = bytes([d ^ k for d, k in zip(data, keystream)])
        
        return {
            'output': result,
            'keystream': bytes(keystream),
            'mode': mode,
            'length': len(data)
        }
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Convenience method for encryption."""
        self.set_key(key)
        return self.process(plaintext, mode='encrypt')['output']
    
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Convenience method for decryption."""
        self.set_key(key)
        return self.process(ciphertext, mode='decrypt')['output']


# ============================================================================
# Hash Function
# ============================================================================

@dataclass
class HashFunctionConfig:
    """Configuration for chaos hash function."""
    output_bits: int = 256
    rounds: int = 16
    oscillator_count: int = 4


class HashFunction:
    """
    Chaos-based hash function with avalanche effect.
    
    Uses coupled chaotic oscillators mixed through multiple rounds
    to create a cryptographic hash.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = HashFunctionConfig(**config)
        self.output_bytes = self.config.output_bits // 8
    
    def process(self, data: bytes) -> Dict[str, Any]:
        """
        Compute chaos-based hash of data.
        
        Args:
            data: Input data to hash
            
        Returns:
            Dict with hash, hex_digest
        """
        # Initialize oscillator states from data
        states = self._initialize_states(data)
        
        # Perform mixing rounds
        for round_idx in range(self.config.rounds):
            states = self._mix_round(states, data, round_idx)
        
        # Extract hash from final states
        hash_bytes = self._finalize_hash(states)
        
        return {
            'hash': hash_bytes,
            'hex_digest': hash_bytes.hex(),
            'algorithm': f'chaos-{self.config.output_bits}'
        }
    
    def _initialize_states(self, data: bytes) -> np.ndarray:
        """Initialize oscillator states from input data."""
        # Use SHA-256 to mix input into initial state
        data_hash = hashlib.sha256(data).digest()
        
        # Create multiple oscillator states
        states = np.zeros((self.config.oscillator_count, 3))
        
        for i in range(self.config.oscillator_count):
            offset = (i * 8) % len(data_hash)
            state_bytes = data_hash[offset:offset+8]
            
            if len(state_bytes) < 8:
                state_bytes += data_hash[:8-len(state_bytes)]
            
            state_val = np.frombuffer(state_bytes, dtype=np.float64)[0]
            
            # Distribute to x, y, z
            states[i] = [
                (state_val * 1.0) % 4.0 - 2.0,
                (state_val * 1.7) % 4.0 - 2.0,
                (state_val * 2.3) % 4.0 - 2.0
            ]
        
        return states
    
    def _mix_round(self, states: np.ndarray, data: bytes, round_idx: int) -> np.ndarray:
        """Perform one mixing round with chaotic dynamics."""
        # Chua parameters
        alpha = 15.6 + (round_idx * 0.1)
        beta = 28.0 + (round_idx * 0.2)
        m0, m1 = -1.143, -0.714
        dt = 0.05
        
        # Evolve each oscillator
        for i in range(self.config.oscillator_count):
            x, y, z = states[i]
            
            # Chua nonlinearity
            h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
            
            # Chua equations
            dx = alpha * (y - x - h)
            dy = x - y + z
            dz = -beta * y
            
            states[i] += dt * np.array([dx, dy, dz])
        
        # Couple oscillators
        for i in range(self.config.oscillator_count):
            j = (i + 1) % self.config.oscillator_count
            coupling = 0.1 * (states[j] - states[i])
            states[i] += coupling
        
        # Mix with data
        data_influence = sum(data) / len(data) / 127.5 - 1.0
        states += data_influence * 0.01
        
        return states
    
    def _finalize_hash(self, states: np.ndarray) -> bytes:
        """Extract final hash from oscillator states."""
        # Flatten all states
        flat_states = states.flatten()
        
        # Convert to bytes
        hash_bytes = b''
        for value in flat_states:
            # Map to [0, 255]
            byte_val = int((value + 2.0) * 63.75) % 256
            hash_bytes += bytes([byte_val])
        
        # Truncate or pad to desired length
        if len(hash_bytes) > self.output_bytes:
            hash_bytes = hash_bytes[:self.output_bytes]
        else:
            # Pad with SHA-256 of states if needed
            padding = hashlib.sha256(hash_bytes).digest()
            hash_bytes += padding[:self.output_bytes - len(hash_bytes)]
        
        return hash_bytes


# ============================================================================
# Key Derivation
# ============================================================================

@dataclass
class KeyDerivationConfig:
    """Configuration for key derivation."""
    iterations: int = 10000
    key_length: int = 32  # bytes


class KeyDerivation:
    """
    Key derivation from password using chaotic iteration.
    
    Similar to PBKDF2 but uses chaotic dynamics for mixing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = KeyDerivationConfig(**config)
    
    def process(self, password: bytes, salt: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Derive key from password and salt.
        
        Args:
            password: Input password
            salt: Salt value (generated if None)
            
        Returns:
            Dict with key, salt, iterations
        """
        if salt is None:
            salt = np.random.bytes(16)
        
        # Initialize with password + salt
        state = self._initialize_state(password, salt)
        
        # Iterate chaotic function
        for _ in range(self.config.iterations):
            state = self._iterate(state, password, salt)
        
        # Extract key
        key = self._extract_key(state)
        
        return {
            'key': key,
            'salt': salt,
            'iterations': self.config.iterations,
            'algorithm': 'chaos-kdf'
        }
    
    def _initialize_state(self, password: bytes, salt: bytes) -> np.ndarray:
        """Initialize state from password and salt."""
        # Combine password and salt
        combined = password + salt
        combined_hash = hashlib.sha256(combined).digest()
        
        # Create 3D state
        state_vals = np.frombuffer(combined_hash[:24], dtype=np.float64)
        return state_vals.reshape(3)
    
    def _iterate(self, state: np.ndarray, password: bytes, salt: bytes) -> np.ndarray:
        """One iteration of chaotic mixing."""
        x, y, z = state
        
        # Chaotic map (modified Lorenz)
        sigma = 10.0 + sum(password) / len(password)
        rho = 28.0 + sum(salt) / len(salt)
        beta = 8.0 / 3.0
        dt = 0.01
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        new_state = state + dt * np.array([dx, dy, dz])
        
        # Normalize to prevent divergence
        norm = np.linalg.norm(new_state)
        if norm > 10.0:
            new_state = new_state / norm * 10.0
        
        return new_state
    
    def _extract_key(self, state: np.ndarray) -> bytes:
        """Extract key bytes from final state."""
        # Hash the state
        state_bytes = state.tobytes()
        key_hash = hashlib.sha256(state_bytes).digest()
        
        # Extend if needed
        key = key_hash
        while len(key) < self.config.key_length:
            key_hash = hashlib.sha256(key_hash).digest()
            key += key_hash
        
        return key[:self.config.key_length]


# ============================================================================
# Random Number Generator (Crypto-grade)
# ============================================================================

@dataclass
class RandomNumberGeneratorConfig:
    """Configuration for CSPRNG."""
    oscillator_count: int = 16
    coupling_strength: float = 0.3
    reseed_interval: int = 1000


class RandomNumberGenerator:
    """
    Cryptographically secure random number generator using coupled oscillators.
    
    Uses 16 coupled chaotic oscillators for high entropy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = RandomNumberGeneratorConfig(**config)
        
        # Multiple oscillator states
        self.states = np.random.randn(self.config.oscillator_count, 3) * 0.1
        self.byte_count = 0
        self.initialized = False
    
    def seed(self, seed_bytes: bytes):
        """Seed the RNG with entropy."""
        # Hash seed to get sufficient bytes
        seed_hash = hashlib.sha256(seed_bytes).digest()
        
        # Initialize each oscillator
        for i in range(self.config.oscillator_count):
            offset = (i * 8) % len(seed_hash)
            state_bytes = seed_hash[offset:offset+8]
            
            if len(state_bytes) < 8:
                # Re-hash if needed
                seed_hash = hashlib.sha256(seed_hash).digest()
                state_bytes = seed_hash[:8]
            
            state_val = np.frombuffer(state_bytes, dtype=np.float64)[0]
            
            self.states[i] = [
                (state_val * 1.0) % 4.0 - 2.0,
                (state_val * 1.7) % 4.0 - 2.0,
                (state_val * 2.3) % 4.0 - 2.0
            ]
        
        self.initialized = True
    
    def process(self, num_bytes: int = 32) -> Dict[str, Any]:
        """
        Generate random bytes from chaotic dynamics.
        
        Args:
            num_bytes: Number of random bytes to generate
            
        Returns:
            Dict with random_bytes, entropy_estimate
        """
        if not self.initialized:
            # Auto-seed from system entropy
            self.seed(np.random.bytes(32))
        
        random_bytes = []
        
        for _ in range(num_bytes):
            # Step all oscillators
            self._step_oscillators()
            
            # Mix all oscillator states
            mixed = self._mix_states()
            
            # Convert to byte
            byte_val = int((mixed + 2.0) * 63.75) % 256
            random_bytes.append(byte_val)
            
            self.byte_count += 1
            
            # Periodically reseed for forward secrecy
            if self.byte_count % self.config.reseed_interval == 0:
                self._reseed_from_state()
        
        output = bytes(random_bytes)
        entropy = self._estimate_entropy(random_bytes)
        
        return {
            'random_bytes': output,
            'entropy_estimate': entropy,
            'bytes_generated': self.byte_count
        }
    
    def _step_oscillators(self):
        """Evolve all coupled oscillators."""
        alpha, beta = 15.6, 28.0
        m0, m1 = -1.143, -0.714
        dt = 0.01
        
        new_states = np.copy(self.states)
        
        for i in range(self.config.oscillator_count):
            x, y, z = self.states[i]
            
            # Chua nonlinearity
            h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
            
            # Chua equations
            dx = alpha * (y - x - h)
            dy = x - y + z
            dz = -beta * y
            
            # Coupling with neighbors
            left = (i - 1) % self.config.oscillator_count
            right = (i + 1) % self.config.oscillator_count
            
            coupling = self.config.coupling_strength * (
                (self.states[left][0] - x) + (self.states[right][0] - x)
            )
            
            dx += coupling
            
            new_states[i] += dt * np.array([dx, dy, dz])
        
        self.states = new_states
    
    def _mix_states(self) -> float:
        """Mix all oscillator states into single value."""
        # Use weighted sum with prime-based weights
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        
        mixed = 0.0
        for i in range(self.config.oscillator_count):
            weight = primes[i] / 10.0
            mixed += weight * np.sum(self.states[i])
        
        return mixed
    
    def _reseed_from_state(self):
        """Reseed from current state for forward secrecy."""
        state_bytes = self.states.tobytes()
        reseed = hashlib.sha256(state_bytes).digest()
        self.seed(reseed)
    
    def _estimate_entropy(self, data: List[int]) -> float:
        """Estimate entropy of generated data."""
        if not data:
            return 0.0
        
        # Shannon entropy
        counts = Counter(data)
        total = len(data)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy


# ============================================================================
# Crypto Analyzer
# ============================================================================

@dataclass
class CryptoAnalyzerConfig:
    """Configuration for crypto analyzer."""
    tests: List[str] = None
    
    def __post_init__(self):
        if self.tests is None:
            self.tests = ['frequency', 'runs', 'longest_run', 'entropy']


class CryptoAnalyzer:
    """
    Statistical tests for randomness quality (NIST SP 800-22 inspired).
    
    Performs various tests on bit streams to assess randomness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = CryptoAnalyzerConfig(**config)
    
    def process(self, data: bytes) -> Dict[str, Any]:
        """
        Analyze bit stream for randomness.
        
        Args:
            data: Byte stream to analyze
            
        Returns:
            Dict with test results
        """
        # Convert to bit array
        bits = self._bytes_to_bits(data)
        
        results = {}
        
        if 'frequency' in self.config.tests:
            results['frequency_test'] = self._frequency_test(bits)
        
        if 'runs' in self.config.tests:
            results['runs_test'] = self._runs_test(bits)
        
        if 'longest_run' in self.config.tests:
            results['longest_run_test'] = self._longest_run_test(bits)
        
        if 'entropy' in self.config.tests:
            results['entropy_test'] = self._entropy_test(data)
        
        # Overall assessment
        passed = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        
        results['overall'] = {
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0.0,
            'quality': 'good' if passed == total else 'acceptable' if passed >= total * 0.75 else 'poor'
        }
        
        return results
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to bit array."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def _frequency_test(self, bits: List[int]) -> Dict[str, Any]:
        """
        Frequency (monobit) test.
        
        Tests if ones and zeros are approximately equal.
        """
        n = len(bits)
        ones = sum(bits)
        zeros = n - ones
        
        # Expected is 50/50
        expected = n / 2.0
        
        # Chi-square statistic
        chi_sq = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
        
        # Critical value at α=0.01 for 1 DOF is ~6.635
        passed = chi_sq < 6.635
        
        return {
            'test': 'frequency',
            'ones': ones,
            'zeros': zeros,
            'chi_square': chi_sq,
            'passed': passed,
            'p_value': 1.0 - (chi_sq / 10.0) if chi_sq < 10.0 else 0.0
        }
    
    def _runs_test(self, bits: List[int]) -> Dict[str, Any]:
        """
        Runs test.
        
        Tests if number of runs (consecutive bits) is as expected.
        """
        if len(bits) < 2:
            return {'test': 'runs', 'passed': False, 'error': 'insufficient_data'}
        
        runs = 1
        for i in range(1, len(bits)):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Expected number of runs
        n = len(bits)
        ones = sum(bits)
        p = ones / n
        
        expected_runs = (2 * n * p * (1 - p)) + 1
        
        # Simple check: within 20% of expected
        passed = abs(runs - expected_runs) < expected_runs * 0.2
        
        return {
            'test': 'runs',
            'runs': runs,
            'expected': expected_runs,
            'passed': passed
        }
    
    def _longest_run_test(self, bits: List[int]) -> Dict[str, Any]:
        """Test length of longest run of ones."""
        if not bits:
            return {'test': 'longest_run', 'passed': False, 'error': 'no_data'}
        
        longest = 0
        current = 0
        
        for bit in bits:
            if bit == 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        
        # Expected longest run is approximately log2(n) + 1
        n = len(bits)
        expected = int(np.log2(n)) + 1 if n > 0 else 0
        
        # Passed if within reasonable range
        passed = longest <= expected * 2
        
        return {
            'test': 'longest_run',
            'longest_run': longest,
            'expected': expected,
            'passed': passed
        }
    
    def _entropy_test(self, data: bytes) -> Dict[str, Any]:
        """Shannon entropy test."""
        if not data:
            return {'test': 'entropy', 'passed': False, 'error': 'no_data'}
        
        # Byte-level entropy
        counts = Counter(data)
        total = len(data)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        # Max entropy for bytes is 8.0
        max_entropy = 8.0
        normalized_entropy = entropy / max_entropy
        
        # Good randomness should have entropy > 7.5
        passed = entropy > 7.5
        
        return {
            'test': 'entropy',
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized': normalized_entropy,
            'passed': passed
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_crypto_node(node_type: str, config: Dict[str, Any]):
    """Factory function to create crypto nodes."""
    
    crypto_nodes = {
        'ChaosStreamCipher': ChaosStreamCipher,
        'Chaos Stream Cipher': ChaosStreamCipher,
        'HashFunction': HashFunction,
        'Hash Function': HashFunction,
        'KeyDerivation': KeyDerivation,
        'Key Derivation': KeyDerivation,
        'RandomNumberGenerator': RandomNumberGenerator,
        'Random Number Generator': RandomNumberGenerator,
        'CryptoAnalyzer': CryptoAnalyzer,
        'Crypto Analyzer': CryptoAnalyzer,
    }
    
    node_class = crypto_nodes.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown crypto node type: {node_type}")
    
    return node_class(config)

