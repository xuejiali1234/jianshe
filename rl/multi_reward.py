from __future__ import annotations

from typing import Any

from .reward import compute_reward


def compute_multi_tls_rewards(
    per_tls_info: dict[str, dict[str, Any]],
    neighbors: dict[str, dict[str, list[str]]],
    switch_applied_by_tls: dict[str, bool],
    weights: dict[str, float],
    reward_mode: str,
    use_prediction_reward: bool,
    prediction_by_tls: dict[str, dict[int, dict[str, Any]]] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float | int | str]], dict[str, float]]:
    prediction_by_tls = prediction_by_tls or {}
    rewards: dict[str, float] = {}
    meta_by_tls: dict[str, dict[str, float | int | str]] = {}
    for tls_id, tls_info in per_tls_info.items():
        local_reward, local_meta = compute_reward(
            tls_info,
            bool(switch_applied_by_tls.get(tls_id)),
            weights,
            reward_mode=reward_mode,
            use_prediction_reward=use_prediction_reward,
            prediction_by_phase=prediction_by_tls.get(tls_id),
        )
        upstream_queue = _neighbor_queue_total(per_tls_info, neighbors.get(tls_id, {}).get("upstream", []))
        downstream_queue = _neighbor_queue_total(per_tls_info, neighbors.get(tls_id, {}).get("downstream", []))
        coordination_penalty = float(
            -weights.get("upstream_spillback", 0.20) * upstream_queue / 50.0
            -weights.get("downstream_pressure", 0.15) * downstream_queue / 50.0
        )
        rewards[tls_id] = float(local_reward + coordination_penalty)
        local_meta["coordination_penalty"] = float(coordination_penalty)
        local_meta["upstream_neighbor_queue"] = float(upstream_queue)
        local_meta["downstream_neighbor_queue"] = float(downstream_queue)
        meta_by_tls[tls_id] = local_meta

    cluster_meta = {
        "mean_reward": float(sum(rewards.values()) / max(len(rewards), 1)),
        "mean_coordination_penalty": float(
            sum(float(meta.get("coordination_penalty", 0.0)) for meta in meta_by_tls.values()) / max(len(meta_by_tls), 1)
        ),
    }
    return rewards, meta_by_tls, cluster_meta


def _neighbor_queue_total(
    per_tls_info: dict[str, dict[str, Any]],
    neighbor_tls_ids: list[str],
) -> float:
    total = 0.0
    for tls_id in neighbor_tls_ids:
        tls_info = per_tls_info.get(str(tls_id), {})
        total += sum(float(item.get("queue_sum", 0.0)) for item in tls_info.get("phase_stats", []))
    return float(total)
