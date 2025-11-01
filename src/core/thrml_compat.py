"""Compatibility helpers to restore rich THRML features absent in PyPI 0.1.3.

This module recreates grouped node helpers (SpinNodes, ContinuousNodes,
DiscreteNodes) and provides EnergyObserver / CorrelationObserver utilities that
mirror the behaviour of the original repository implementations while relying
solely on the public PyPI API documented in ``docs/thrml_docs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from thrml import Block, SpinNode
from thrml.block_management import block_state_to_global
from thrml.models import IsingEBM
from thrml.observers import AbstractObserver


# ---------------------------------------------------------------------------
# Node group compatibility wrappers
# ---------------------------------------------------------------------------


@dataclass
class NodeGroup:
    """Convenience container bundling THRML node instances with metadata."""

    name: str
    node_ids: Tuple[int, ...]
    nodes: Tuple[SpinNode, ...]
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._id_to_node: Dict[int, SpinNode] = dict(zip(self.node_ids, self.nodes))

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> SpinNode:
        return self.nodes[idx]

    def nodes_for_ids(self, ids: Iterable[int]) -> List[SpinNode]:
        return [self._id_to_node[i] for i in ids if i in self._id_to_node]

    def as_block(self) -> Block:
        return Block(self.nodes)


@dataclass
class SpinNodes(NodeGroup):
    """Binary spin node group used for Ising-style variables."""

    def __init__(self, name: str, node_ids: Sequence[int], nodes: Sequence[SpinNode] | None = None):
        node_tuple = tuple(nodes) if nodes is not None else tuple(SpinNode() for _ in node_ids)
        super().__init__(name=name, node_ids=tuple(node_ids), nodes=node_tuple)


@dataclass
class ContinuousNodes(NodeGroup):
    """Continuous node group (approximated with SpinNode carriers)."""

    def __init__(
        self,
        name: str,
        node_ids: Sequence[int],
        min_value: float = -1.0,
        max_value: float = 1.0,
        nodes: Sequence[SpinNode] | None = None,
    ):
        node_tuple = tuple(nodes) if nodes is not None else tuple(SpinNode() for _ in node_ids)
        metadata = {"min_value": float(min_value), "max_value": float(max_value)}
        super().__init__(name=name, node_ids=tuple(node_ids), nodes=node_tuple, metadata=metadata)


@dataclass
class DiscreteNodes(NodeGroup):
    """Discrete node group represented via binary spins plus metadata."""

    def __init__(
        self,
        name: str,
        node_ids: Sequence[int],
        n_values: int = 4,
        nodes: Sequence[SpinNode] | None = None,
    ):
        node_tuple = tuple(nodes) if nodes is not None else tuple(SpinNode() for _ in node_ids)
        metadata = {"n_values": int(n_values)}
        super().__init__(name=name, node_ids=tuple(node_ids), nodes=node_tuple, metadata=metadata)


# ---------------------------------------------------------------------------
# Observer compatibility layer
# ---------------------------------------------------------------------------


class _IsingStateExtractor(eqx.Module):
    """Helper to convert block states into a dense spin vector."""

    node_index: Dict = eqx.field(static=True)

    def extract(self, program, state_free, state_clamped) -> jnp.ndarray:
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        spins = jnp.zeros(len(self.node_index), dtype=jnp.float32)

        for block, values in zip(program.gibbs_spec.blocks, global_state):
            block_array = jnp.asarray(values).astype(jnp.float32)
            block_spins = block_array * 2.0 - 1.0
            for idx, node in enumerate(block):
                node_pos = self.node_index.get(node)
                if node_pos is not None:
                    spins = spins.at[node_pos].set(block_spins[idx])

        return spins


class EnergyObserver(AbstractObserver):
    """Observer that records the instantaneous Ising energy per sample."""

    _extractor: _IsingStateExtractor
    _biases: jnp.ndarray
    _beta: float
    _edge_indices: jnp.ndarray
    _edge_weights: jnp.ndarray

    def __init__(self, model: IsingEBM):
        node_index = {node: i for i, node in enumerate(model.nodes)}
        edge_indices = jnp.array(
            [
                (node_index[edge[0]], node_index[edge[1]])
                for edge in model.edges
            ],
            dtype=jnp.int32,
        ) if model.edges else jnp.zeros((0, 2), dtype=jnp.int32)

        object.__setattr__(self, "_extractor", _IsingStateExtractor(node_index))
        object.__setattr__(self, "_biases", jnp.asarray(model.biases).astype(jnp.float32))
        object.__setattr__(self, "_beta", float(model.beta))
        object.__setattr__(self, "_edge_indices", edge_indices)
        object.__setattr__(self, "_edge_weights", jnp.asarray(model.weights).astype(jnp.float32))

    def init(self):
        return None

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        spins = self._extractor.extract(program, state_free, state_clamped)

        field_energy = jnp.dot(self._biases, spins)

        if self._edge_indices.shape[0] > 0:
            spin_i = spins[self._edge_indices[:, 0]]
            spin_j = spins[self._edge_indices[:, 1]]
            interaction_energy = jnp.sum(self._edge_weights * spin_i * spin_j)
        else:
            interaction_energy = 0.0

        energy = -self._beta * (field_energy + interaction_energy)
        return carry, jnp.asarray(energy, dtype=jnp.float32)


class CorrelationObserver(AbstractObserver):
    """Observer computing pairwise correlations for specified node pairs."""

    _extractor: _IsingStateExtractor
    _pair_indices: jnp.ndarray

    def __init__(self, model: IsingEBM, node_pairs: Sequence[Tuple] | None = None):
        node_index = {node: i for i, node in enumerate(model.nodes)}
        if node_pairs is None:
            node_pairs = model.edges

        pair_indices = jnp.array(
            [
                (node_index[pair[0]], node_index[pair[1]])
                for pair in node_pairs
            ],
            dtype=jnp.int32,
        ) if node_pairs else jnp.zeros((0, 2), dtype=jnp.int32)

        object.__setattr__(self, "_extractor", _IsingStateExtractor(node_index))
        object.__setattr__(self, "_pair_indices", pair_indices)

    def init(self):
        return None

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        spins = self._extractor.extract(program, state_free, state_clamped)

        if self._pair_indices.shape[0] == 0:
            correlations = jnp.zeros((0,), dtype=jnp.float32)
        else:
            spin_i = spins[self._pair_indices[:, 0]]
            spin_j = spins[self._pair_indices[:, 1]]
            correlations = spin_i * spin_j

        return carry, correlations


__all__ = [
    "SpinNodes",
    "ContinuousNodes",
    "DiscreteNodes",
    "EnergyObserver",
    "CorrelationObserver",
]

