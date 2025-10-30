"""Performance benchmarking utilities for the GMCS platform.

This module provides a CLI entry-point (python -m src.tools.perf) that measures
simulation throughput (steps/second) for different node counts and grid sizes.

Usage examples:
    python -m src.tools.perf
    python -m src.tools.perf --steps 200 --warmup 25
    python -m src.tools.perf --configs 64:96,256:128
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List

import jax
import jax.numpy as jnp

from src.core.state import initialize_system_state
from src.core.simulation import simulation_step


@dataclass
class BenchmarkConfig:
    """Parameterisation of a benchmark run."""

    n_nodes: int
    grid_size: int

    @classmethod
    def parse_many(cls, raw: str) -> List["BenchmarkConfig"]:
        """Parse comma-separated configs in the form ``<nodes>:<grid>``."""

        configs: List[BenchmarkConfig] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                nodes_str, grid_str = token.split(":")
                configs.append(cls(int(nodes_str), int(grid_str)))
            except ValueError as exc:  # pragma: no cover - CLI validation
                raise argparse.ArgumentTypeError(
                    "Configurations must be formatted as <nodes>:<grid>."
                ) from exc
        return configs


def _prepare_state(config: BenchmarkConfig, seed: int = 0) -> jax.Array:
    """Create an initial state tailored to the benchmark configuration."""

    n_nodes = config.n_nodes
    grid = config.grid_size
    state = initialize_system_state(
        jax.random.PRNGKey(seed),
        n_max=max(n_nodes, 1),
        grid_w=grid,
        grid_h=grid,
    )

    # Activate the first ``n_nodes`` oscillators and distribute them in a grid.
    mask = jnp.zeros_like(state.node_active_mask).at[:n_nodes].set(1.0)

    side = int(jnp.ceil(jnp.sqrt(n_nodes)))
    xs = jnp.linspace(0.0, grid - 1.0, side)
    ys = jnp.linspace(0.0, grid - 1.0, side)
    xx, yy = jnp.meshgrid(xs, ys)
    coords = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
    coords = coords[:n_nodes]
    positions = state.node_positions.at[:n_nodes].set(coords)

    # Small random perturbation for oscillators to avoid a completely flat start.
    perturb = jax.random.normal(jax.random.PRNGKey(seed + 1), (n_nodes, 3)) * 0.01
    oscillator_state = state.oscillator_state.at[:n_nodes].set(perturb)

    return state._replace(
        node_active_mask=mask,
        node_positions=positions,
        oscillator_state=oscillator_state,
    )


def _run_steps(state, steps: int, *, enable_ebm: bool) -> jax.Array:
    """Run ``steps`` simulation iterations and return the final state."""

    for _ in range(steps):
        state = simulation_step(state, enable_ebm_feedback=enable_ebm)
    return state


def benchmark_configuration(
    config: BenchmarkConfig,
    *,
    warmup_steps: int,
    measure_steps: int,
    enable_ebm: bool,
) -> dict:
    """Benchmark one configuration and return timing metadata."""

    state = _prepare_state(config)

    # Warmup: compile and stabilise caches.
    state = _run_steps(state, warmup_steps, enable_ebm=enable_ebm)
    jax.block_until_ready(state.field_p)

    # Measure throughput.
    start = time.perf_counter()
    state = _run_steps(state, measure_steps, enable_ebm=enable_ebm)
    jax.block_until_ready(state.field_p)
    elapsed = time.perf_counter() - start

    steps_per_second = measure_steps / elapsed if elapsed > 0 else float("inf")
    milliseconds_per_step = (elapsed / measure_steps) * 1000.0 if measure_steps else 0.0

    return {
        "nodes": config.n_nodes,
        "grid": f"{config.grid_size}×{config.grid_size}",
        "steps": measure_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_second,
        "ms_per_step": milliseconds_per_step,
    }


def default_configs() -> Iterable[BenchmarkConfig]:
    """Return the default suite of benchmark configurations."""

    return (
        BenchmarkConfig(64, 96),
        BenchmarkConfig(256, 128),
        BenchmarkConfig(512, 192),
        BenchmarkConfig(1024, 256),
    )


def format_results(rows: Iterable[dict]) -> str:
    """Format benchmark results into a table string."""

    header = "Configuration            Steps   Steps/s   ms/step"
    lines = [header, "-" * len(header)]
    for row in rows:
        cfg = f"{row['nodes']:>4} nodes, {row['grid']}"
        lines.append(
            f"{cfg:<24} {row['steps']:>6d}   {row['steps_per_sec']:>8.1f}   {row['ms_per_step']:>7.3f}"
        )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GMCS performance benchmarks")
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma separated <nodes>:<grid> entries (e.g., 64:96,256:128).",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup steps before measuring.")
    parser.add_argument("--steps", type=int, default=200, help="Measured steps per configuration.")
    parser.add_argument(
        "--no-ebm",
        action="store_true",
        help="Disable EBM feedback during benchmarks for raw physics performance.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    configs = (
        BenchmarkConfig.parse_many(args.configs)
        if args.configs
        else list(default_configs())
    )

    backend = jax.default_backend()
    device = jax.devices()[0]
    print(f"JAX backend: {backend} ({device})")
    print(
        f"Warmup: {args.warmup} steps | Measured: {args.steps} steps | EBM feedback: {'off' if args.no_ebm else 'on'}"
    )

    results = [
        benchmark_configuration(
            config,
            warmup_steps=args.warmup,
            measure_steps=args.steps,
            enable_ebm=not args.no_ebm,
        )
        for config in configs
    ]

    print()
    print(format_results(results))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
