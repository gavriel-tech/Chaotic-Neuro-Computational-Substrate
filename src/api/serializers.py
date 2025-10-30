"""
Binary serialization for WebSocket transmission with adaptive optimisation.

Efficient binary packet format for streaming simulation state to the frontend.
"""

from __future__ import annotations

import struct
import zlib
from typing import Tuple

import numpy as np
import jax.numpy as jnp
from jax import image

from src.core.state import SystemState, N_MAX


_HEADER_STRUCT = struct.Struct("<7If")
_FLAG_COMPRESSED = 1 << 0


def _auto_downsample_resolution(
    base_w: int,
    base_h: int,
    active_nodes: int,
) -> Tuple[int, int]:
    """Dynamically reduce resolution when many nodes are active."""

    if active_nodes <= 256:
        factor = 1
    elif active_nodes <= 512:
        factor = 2
    else:
        factor = 4

    down_w = max(32, base_w // factor)
    down_h = max(32, base_h // factor)
    return down_w, down_h


def serialize_for_frontend(
    state: SystemState,
    down_w: int = 128,
    down_h: int = 128,
    *,
    compression: str = "auto",
    max_nodes: int | None = 512,
    mask_threshold: float = 0.1,
) -> bytes:
    """Serialize state into an optimised binary packet for WebSocket streaming.

    Packet header is 7 × uint32 (little-endian):
        - downsampled width/height
        - active node count
        - total capacity (N_MAX)
        - flags (bit0: payload compressed)
        - uncompressed payload size (bytes)
        - stored payload size (bytes)

    Payload layout (uncompressed):
        - Field data    : float32[down_w * down_h]
        - Oscillator    : float32[active_nodes * 3]
        - Positions     : float32[active_nodes * 3]
        - Mask          : float32[active_nodes]
    """

    if hasattr(state.field_p, "block_until_ready"):
        state.field_p.block_until_ready()

    field = np.asarray(state.field_p)
    osc_state = np.asarray(state.oscillator_state)
    positions = np.asarray(state.node_positions)
    mask = np.asarray(state.node_active_mask)

    active_indices = np.nonzero(mask > mask_threshold)[0]
    active_count = int(active_indices.size)

    if max_nodes is not None and active_count > max_nodes:
        stride = max(1, active_count // max_nodes)
        active_indices = active_indices[::stride][:max_nodes]
        active_count = int(active_indices.size)

    if active_count == 0:
        active_indices = np.zeros(0, dtype=np.int64)

    # Adaptive field resolution scaling.
    down_w_eff, down_h_eff = _auto_downsample_resolution(down_w, down_h, active_count)

    if down_w_eff != state.field_p.shape[0] or down_h_eff != state.field_p.shape[1]:
        field_downsampled = image.resize(state.field_p, (down_w_eff, down_h_eff), method="linear")
        field_array = np.asarray(field_downsampled, dtype=np.float32)
    else:
        field_array = field.astype(np.float32, copy=False)

    osc_active = osc_state[active_indices].astype(np.float32, copy=False)

    # Promote 2D positions to 3D (x, y=0, z) for instanced rendering convenience.
    pos_active = positions[active_indices].astype(np.float32, copy=False)
    if pos_active.size:
        zeros = np.zeros((pos_active.shape[0], 1), dtype=np.float32)
        pos_3d = np.concatenate([pos_active[:, :1], zeros, pos_active[:, 1:2]], axis=1)
    else:
        pos_3d = np.zeros((0, 3), dtype=np.float32)

    mask_active = mask[active_indices].astype(np.float32, copy=False)

    field_bytes = field_array.ravel().tobytes(order="C")
    osc_bytes = osc_active.reshape(-1).tobytes(order="C")
    pos_bytes = pos_3d.reshape(-1).tobytes(order="C")
    mask_bytes = mask_active.reshape(-1).tobytes(order="C")

    raw_payload = field_bytes + osc_bytes + pos_bytes + mask_bytes
    uncompressed_size = len(raw_payload)

    flags = 0
    payload = raw_payload

    should_compress = False
    if compression == "always":
        should_compress = True
    elif compression == "auto" and uncompressed_size > 600_000:
        should_compress = True

    if should_compress and uncompressed_size > 0:
        compressed = zlib.compress(raw_payload, level=3)
        if len(compressed) < uncompressed_size:
            payload = compressed
            flags |= _FLAG_COMPRESSED

    header = _HEADER_STRUCT.pack(
        down_w_eff,
        down_h_eff,
        active_count,
        N_MAX,
        flags,
        uncompressed_size,
        len(payload),
        float(state.t[0] if state.t.ndim else state.t),
    )

    return header + payload


def deserialize_packet_info(packet: bytes) -> dict:
    """Inspect packet header without full deserialisation."""

    if len(packet) < _HEADER_STRUCT.size:
        raise ValueError(f"Packet too short: {len(packet)} bytes")

    down_w, down_h, active_count, total_capacity, flags, payload_size, stored_size, sim_time = _HEADER_STRUCT.unpack_from(
        packet
    )

    return {
        "down_w": int(down_w),
        "down_h": int(down_h),
        "active_count": int(active_count),
        "capacity": int(total_capacity),
        "compressed": bool(flags & _FLAG_COMPRESSED),
        "payload_size": int(payload_size),
        "stored_size": int(stored_size),
        "packet_size": len(packet),
        "simulation_time": float(sim_time),
    }


def get_javascript_deserializer() -> str:
    """Generate reference JavaScript for deserialising the binary packet."""

    return """
const HEADER_UINT32_COUNT = 7;
const HEADER_FLOAT_COUNT = 1;
const HEADER_BYTES = (HEADER_UINT32_COUNT + HEADER_FLOAT_COUNT) * 4;
const FLAG_COMPRESSED = 1 << 0;

function deserializeBinaryPacket(arrayBuffer) {
    const header = new DataView(arrayBuffer, 0, HEADER_BYTES);
    const downW = header.getUint32(0, true);
    const downH = header.getUint32(1 * 4, true);
    const activeCount = header.getUint32(2 * 4, true);
    const capacity = header.getUint32(3 * 4, true);
    const flags = header.getUint32(4 * 4, true);
    const payloadSize = header.getUint32(5 * 4, true);
    const storedSize = header.getUint32(6 * 4, true);
    const simTime = header.getFloat32(7 * 4, true);

    let payload = new Uint8Array(arrayBuffer, HEADER_BYTES, storedSize);
    if (flags & FLAG_COMPRESSED) {
        payload = fflate.decompressSync(payload, new Uint8Array(payloadSize));
    } else if (payload.length !== payloadSize) {
        payload = payload.slice(0, payloadSize);
    }

    let offset = 0;
    const fieldCount = downW * downH;
    const field = new Float32Array(payload.buffer, payload.byteOffset + offset, fieldCount);
    offset += fieldCount * 4;

    const osc = new Float32Array(payload.buffer, payload.byteOffset + offset, activeCount * 3);
    offset += activeCount * 3 * 4;

    const positions = new Float32Array(payload.buffer, payload.byteOffset + offset, activeCount * 3);
    offset += activeCount * 3 * 4;

    const mask = new Float32Array(payload.buffer, payload.byteOffset + offset, activeCount);

    return {
        downW,
        downH,
        activeCount,
        capacity,
        field: new Float32Array(field),
        oscillators: new Float32Array(osc),
        positions: new Float32Array(positions),
        mask: new Float32Array(mask),
        simTime,
    };
}
"""


def estimate_bandwidth(
    fps: int = 30,
    down_w: int = 128,
    down_h: int = 128,
    active_nodes: int = N_MAX,
) -> dict:
    """Estimate network bandwidth requirements (uncompressed)."""

    header_size = _HEADER_STRUCT.size
    field_size = down_w * down_h * 4
    osc_size = active_nodes * 3 * 4
    pos_size = active_nodes * 3 * 4
    mask_size = active_nodes * 4
    payload_size = field_size + osc_size + pos_size + mask_size
    packet_size = header_size + payload_size

    bytes_per_second = packet_size * fps

    return {
        "packet_size_bytes": packet_size,
        "packet_size_kb": packet_size / 1024,
        "fps": fps,
        "bandwidth_bytes_per_sec": bytes_per_second,
        "bandwidth_kb_per_sec": bytes_per_second / 1024,
        "bandwidth_mb_per_sec": bytes_per_second / (1024 * 1024),
        "bandwidth_mbps": (bytes_per_second * 8) / (1024 * 1024),
    }

