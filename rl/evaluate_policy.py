from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .baselines import make_policy
from .env import SignalControlEnv, PROJECT_ROOT


def evaluate_policy(
    config_path: str | Path,
    policy_name: str,
    sim_end: int | None,
    out_path: str | Path,
    seed: int | None = None,
) -> dict[str, float | int | str]:
    env = SignalControlEnv(config_path)
    if sim_end is not None:
        env.episode_s = int(sim_end)
    policy = make_policy(policy_name)
    rows = []
    total_reward = 0.0
    switch_count = 0
    observation = env.reset(seed=seed)
    done = False
    try:
        while not done:
            action = policy.act(observation, env.last_info)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            switch_count += 1 if info.get("switch_applied") else 0
            queue_sum = sum(float(item.get("queue_sum", 0.0)) for item in info.get("phase_stats", []))
            arrival_sum = sum(float(item.get("arrival_flow_sum", 0.0)) for item in info.get("phase_stats", []))
            rows.append(
                {
                    "policy": policy.name,
                    "sim_time_s": round(float(info.get("sim_time_s", 0.0)), 3),
                    "reward": round(float(reward), 6),
                    "queue_sum": round(queue_sum, 4),
                    "arrival_flow_sum": round(arrival_sum, 4),
                    "vehicle_count": int(info.get("vehicle_count", 0)),
                    "mean_speed_mps": round(float(info.get("mean_speed_mps", 0.0)), 4),
                    "switch_applied": int(bool(info.get("switch_applied"))),
                    "current_phase": int(info.get("current_phase", -1)),
                }
            )
    finally:
        env.close()

    destination = Path(out_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["policy"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "policy": policy.name,
        "steps": len(rows),
        "total_reward": round(float(total_reward), 6),
        "mean_reward": round(float(total_reward / max(len(rows), 1)), 6),
        "switch_count": switch_count,
        "mean_queue": round(sum(row["queue_sum"] for row in rows) / max(len(rows), 1), 4) if rows else 0.0,
        "mean_speed_mps": round(sum(row["mean_speed_mps"] for row in rows) / max(len(rows), 1), 4) if rows else 0.0,
        "out": str(destination),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single-intersection signal control policy.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_signal_config.json"))
    parser.add_argument("--policy", choices=["webster", "max_pressure"], default="webster")
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    summary = evaluate_policy(args.config, args.policy, args.sim_end, args.out, args.seed)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
