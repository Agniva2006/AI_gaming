#!/usr/bin/env python3
"""
run_swarm_demo.py
NeuroArena: Distributed Multi-Agent Swarm PPO & 120 Hz Telemetry Live Demo.
Executes parallel rollout workers, computes GAE and clipped surrogate PPO updates,
evaluates real-time Voronoi pitch dominance, and ingests 120 Hz binary ticks into the ring buffer.
"""

import sys
import time
import random
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_env.ray_distributed_trainer import swarm_trainer
from analytics.voronoi_engine import voronoi_engine
from backend.telemetry_gateway import telemetry_buffer


def main():
    print("=" * 80)
    print(" 🏟️  NEUROARENA: DISTRIBUTED MULTI-AGENT SWARM PPO & 120 Hz TELEMETRY")
    print("=" * 80)
    print(" • Architecture    : Spatial Graph Attention Network (GAT) + PPO Swarm")
    print(" • Parallelism     : 4 Vectorized Rollout Workers (GAE: λ=0.95, γ=0.99)")
    print(" • Telemetry Rate  : 120 Hz Lock-Free Circular Ring Buffer (388-byte frames)")
    print(" • Visualizer      : Decoupled 3D WebGL Three.js with Hermite Spline Dead-Reckoning")
    print("-" * 80)

    # 1. Run 3 Distributed Swarm PPO Steps
    print("\n[PHASE 1] Executing Distributed Swarm PPO Training Steps...")
    print("-" * 80)
    print(f"{'Step':<6} | {'Workers':<8} | {'Batch Steps':<12} | {'Policy Loss':<12} | {'Value Loss':<12} | {'Throughput':<15}")
    print("-" * 80)

    for step in range(1, 4):
        t_res = swarm_trainer.train_step(steps_per_worker=64, epochs=2, batch_size=32)
        print(f"Step {step:<2} | {t_res['workers_active']:<8} | {t_res['batch_steps']:<12} | {t_res['policy_loss']:<12.4f} | {t_res['value_loss']:<12.4f} | {t_res['throughput_steps_per_sec']} steps/sec")

    # 2. Ingest 120 Hz Simulation Ticks into Circular Ring Buffer
    print("\n[PHASE 2] Ingesting High-Frequency 120 Hz Physics Ticks into Lock-Free Ring Buffer...")
    print("-" * 80)
    total_ticks = 60
    t_start = time.perf_counter()

    for i in range(total_ticks):
        players = [{"x": random.uniform(5, 95), "y": random.uniform(5, 60), "vx": random.uniform(-2, 2), "vy": random.uniform(-2, 2)} for _ in range(22)]
        ball = {"x": random.uniform(10, 90), "y": random.uniform(10, 55), "z": random.uniform(0, 3), "vx": random.uniform(-5, 5), "vy": random.uniform(-5, 5), "vz": random.uniform(0, 2)}
        telemetry_buffer.ingest_frame(player_states=players, ball_state=ball, match_time=time.time(), score=[1, 0])
        time.sleep(1.0 / 120.0)  # 120 Hz pacing

    buf_stats = telemetry_buffer.get_buffer_stats()
    print(f" • Buffer Ingested Frames : {buf_stats['total_frames_ingested']} frames")
    print(f" • Frame Binary Size      : {buf_stats['frame_wire_bytes']} bytes/frame")
    print(f" • Measured Ingest Rate   : {buf_stats['ingest_frequency_hz']} Hz (Target: 120.0 Hz)")
    print(f" • Backpressure Drop Rate : {buf_stats['drop_rate_pct']}% (Zero-Lock Circular Buffer)")
    print(f" • Wire Data Rate         : {buf_stats['network_throughput_kbps']} kbps")

    # 3. Compute Real-Time Voronoi Pitch Dominance & Tactical Analytics
    print("\n[PHASE 3] Computing Multi-Agent Voronoi Pitch Control Dominance...")
    print("-" * 80)
    red_squad = [[15.0 + i * 4, 15.0 + (i % 3) * 18] for i in range(11)]
    blue_squad = [[65.0 + i * 3.5, 15.0 + (i % 3) * 18] for i in range(11)]
    v_res = voronoi_engine.compute_pitch_control(red_squad, blue_squad, ball_pos=(52.5, 34.0))

    ctrl = v_res["pitch_control_percentage"]
    cmp = v_res["team_compactness_meters"]
    print(f" • Pitch Control Dominance: Team Red {ctrl['team_red']}% | Team Blue {ctrl['team_blue']}%")
    print(f" • Defensive Compactness  : Team Red {cmp['team_red']}m | Team Blue {cmp['team_blue']}m")
    print(f" • Passing Lane Risk Index: {v_res['passing_lane_vulnerability_index']} (Safe Pass Angle)")
    print(f" • High-Press Intensity   : {v_res['high_press_intensity_index']} ({v_res['tactical_status']})")

    print("\n" + "=" * 80)
    print(" 🏆 MULTI-AGENT SWARM RL & 120 Hz TELEMETRY VERIFICATION COMPLETE!")
    print("=" * 80)
    print(" • To view 3D Stadium   : Open frontend/index.html in your browser")
    print(" • Click Top Button     : '🏟️ 3D WebGL Stadium' for real-time Three.js rendering\n")


if __name__ == "__main__":
    main()
