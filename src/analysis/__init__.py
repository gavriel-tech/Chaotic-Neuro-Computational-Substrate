"""
Analysis module for GMCS.

Provides real-time signal analysis nodes for the node graph system.
"""

from .analysis_nodes import (
    FFTAnalyzer,
    FFTAnalyzerConfig,
    PatternRecognizer,
    PatternRecognizerConfig,
    LyapunovCalculator,
    LyapunovConfig,
    AttractorAnalyzer,
    AttractorAnalyzerConfig,
    create_analysis_node
)

__all__ = [
    'FFTAnalyzer',
    'FFTAnalyzerConfig',
    'PatternRecognizer',
    'PatternRecognizerConfig',
    'LyapunovCalculator',
    'LyapunovConfig',
    'AttractorAnalyzer',
    'AttractorAnalyzerConfig',
    'create_analysis_node'
]

