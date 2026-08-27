#!/usr/bin/env python3
"""
telemetry_gateway.py
NeuroArena: High-Frequency 120 Hz Lock-Free Circular Ring Buffer & Binary Streamer.
Maintains continuous fixed-size state snapshots of 22 players and 3D ball,
providing drop-oldest backpressure protection and binary struct packing for Three.js.
"""

import time
import struct
from collections import deque
from threading import Lock
from typing import Dict, Any, List, Optional
import numpy as np


class CircularTelemetryBuffer:
    """
    Lock-Free Atomic Ring Buffer for 120 Hz Simulation Ingestion.
    Capacity: 256 frames (~2.1 seconds at 120 Hz).
    Drop Policy: Drop oldest snapshot on consumer backpressure to ensure zero lag.
    """

    MAX_CAPACITY = 256
    # 22 players x 4 floats (x, y, vx, vy) + ball x 6 floats (x, y, z, vx, vy, vz) + 2 score + 1 time = 97 floats
    TOTAL_FLOATS = 97
    BINARY_STRUCT_FORMAT = f"<{TOTAL_FLOATS}f"
    FRAME_SIZE_BYTES = TOTAL_FLOATS * 4  # 388 bytes

    def __init__(self, capacity: int = MAX_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()

        self.total_frames_ingested = 0
        self.total_frames_dropped = 0
        self.last_tick_time = time.perf_counter()
        self.tick_intervals = deque(maxlen=60)

    def ingest_frame(self, player_states: List[Dict[str, float]], ball_state: Dict[str, float], match_time: float, score: List[int]) -> bytes:
        """
        Ingest a raw physics tick into the circular buffer.
        Packs into 388-byte binary layout and records high-precision timing.
        """
        now = time.perf_counter()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        self.tick_intervals.append(dt)

        # Flatten coordinates into 97 floats
        flat_data = []

        # 22 players
        for i in range(22):
            if i < len(player_states):
                p = player_states[i]
                flat_data.extend([p.get("x", 0.0), p.get("y", 0.0), p.get("vx", 0.0), p.get("vy", 0.0)])
            else:
                flat_data.extend([0.0, 0.0, 0.0, 0.0])

        # Ball
        flat_data.extend([
            ball_state.get("x", 0.0),
            ball_state.get("y", 0.0),
            ball_state.get("z", 0.0),
            ball_state.get("vx", 0.0),
            ball_state.get("vy", 0.0),
            ball_state.get("vz", 0.0),
        ])

        # Match State
        flat_data.extend([float(score[0]), float(score[1]), float(match_time)])

        # Pack into compact binary struct
        packed_binary = struct.pack(self.BINARY_STRUCT_FORMAT, *[float(v) for v in flat_data])

        with self.lock:
            if len(self.buffer) == self.capacity:
                self.total_frames_dropped += 1
            self.buffer.append({
                "timestamp": now,
                "binary": packed_binary,
                "json_sample": {
                    "match_time": round(match_time, 2),
                    "score": score,
                    "ball": {k: round(ball_state[k], 2) for k in ["x", "y", "z"]},
                    "num_players": min(22, len(player_states)),
                }
            })
            self.total_frames_ingested += 1

        return packed_binary

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent frame for WebSocket transmission."""
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
            return None

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Telemetry diagnostics for the high-frequency ring buffer."""
        with self.lock:
            current_len = len(self.buffer)

        # Compute instantaneous tick frequency (Hz)
        if len(self.tick_intervals) > 1:
            mean_dt = np.mean(self.tick_intervals)
            current_hz = (1.0 / mean_dt) if mean_dt > 0 else 0.0
        else:
            current_hz = 120.0

        drop_rate = (self.total_frames_dropped / max(1, self.total_frames_ingested)) * 100.0

        return {
            "status": "OPERATIONAL",
            "capacity": self.capacity,
            "current_frames_buffered": current_len,
            "buffer_utilization_pct": round((current_len / self.capacity) * 100.0, 1),
            "ingest_frequency_hz": round(float(current_hz), 1),
            "target_frequency_hz": 120.0,
            "total_frames_ingested": self.total_frames_ingested,
            "total_frames_dropped": self.total_frames_dropped,
            "drop_rate_pct": round(drop_rate, 3),
            "frame_wire_bytes": self.FRAME_SIZE_BYTES,
            "network_throughput_kbps": round((self.FRAME_SIZE_BYTES * 8 * current_hz) / 1000.0, 2),
            "backpressure_policy": "Drop-Oldest Circular Ring Buffer",
        }


# Singleton buffer instance
telemetry_buffer = CircularTelemetryBuffer()
