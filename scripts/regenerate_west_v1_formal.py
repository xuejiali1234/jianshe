from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEST_DIR = ROOT / "reports" / "rl_signal_control_multi" / "west_v1"
SCAN_DIR = ROOT / "reports" / "rl_signal_control_multi" / "west_v1_checkpoint_scan"
OUT_DIR = ROOT / "reports" / "rl_signal_control_multi" / "west_v1_formal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "default": "默认运行态",
    "incident": "事故场景",
    "control": "配时扰动场景",
}

SOURCES = {
    "MaxPressure": {
        "default": WEST_DIR / "default_west_max_pressure_eval.csv",
        "incident": WEST_DIR / "incident_west_max_pressure_eval.csv",
        "control": WEST_DIR / "control_west_max_pressure_eval.csv",
    },
    "DQN-no-pred-west-5000": {
        "default": WEST_DIR / "default_west_multi_no_pred_v1_eval.csv",
        "incident": WEST_DIR / "incident_west_multi_no_pred_v1_eval.csv",
        "control": WEST_DIR / "control_west_multi_no_pred_v1_eval.csv",
    },
    "DQN-pred-v1-west-2000": {
        "default": SCAN_DIR / "west_multi_pred_v1_2000_default.csv",
        "incident": SCAN_DIR / "west_multi_pred_v1_2000_incident.csv",
        "control": SCAN_DIR / "west_multi_pred_v1_2000_control.csv",
    },
}

POLICIES = ["MaxPressure", "DQN-no-pred-west-5000", "DQN-pred-v1-west-2000"]
COLORS = ["#7b8794", "#00c2ff", "#ff8c42"]


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def summarize(rows: list[dict[str, str]], scenario: str) -> dict[str, float | str]:
    rewards = [to_float(r["reward"]) for r in rows]
    queues = [to_float(r["queue_sum"]) for r in rows]
    speeds = [to_float(r["mean_speed_mps"]) * 3.6 for r in rows]
    switches = [int(float(r["switch_count"])) for r in rows]
    ready = [int(float(r["prediction_ready"])) for r in rows]
    times = [to_float(r["sim_time_s"]) for r in rows]

    summary: dict[str, float | str] = {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_queue": sum(queues) / len(queues),
        "max_queue": max(queues),
        "mean_speed_kmh": sum(speeds) / len(speeds),
        "switch_count": sum(switches),
        "prediction_ready_share": sum(ready) / len(ready),
        "first_switch_after_event_s": "",
        "pre_event_mean_queue": sum(queues) / len(queues),
        "post_event_mean_queue": sum(queues) / len(queues),
        "post_event_first_300s_mean_queue": sum(queues) / len(queues),
    }

    if scenario in {"incident", "control"}:
        event_s = 1200.0
        post_limit = 1500.0
        pre = [queues[i] for i, t in enumerate(times) if t < event_s]
        post = [queues[i] for i, t in enumerate(times) if t >= event_s]
        post300 = [queues[i] for i, t in enumerate(times) if event_s <= t < post_limit]
        if pre:
            summary["pre_event_mean_queue"] = sum(pre) / len(pre)
        if post:
            summary["post_event_mean_queue"] = sum(post) / len(post)
        if post300:
            summary["post_event_first_300s_mean_queue"] = sum(post300) / len(post300)
        for i, t in enumerate(times):
            if t >= event_s and switches[i] > 0:
                summary["first_switch_after_event_s"] = t
                break

    return summary


def write_table(summary_rows: list[dict[str, float | str]]) -> None:
    csv_path = OUT_DIR / "policy_comparison_west_v1_formal.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = OUT_DIR / "policy_comparison_west_v1_formal.md"
    lines = [
        "# 西侧 3 路口联动控制正式对比",
        "",
        "## 指标表",
        "",
        "| 场景 | 策略 | mean_queue | max_queue | mean_speed_kmh | mean_reward | switch_count | post_event_first_300s_mean_queue | first_switch_after_event_s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in ["default", "incident", "control"]:
        for policy in POLICIES:
            row = next(r for r in summary_rows if r["scenario"] == scenario and r["policy"] == policy)
            lines.append(
                f"| {row['scenario_label']} | {policy} | {row['mean_queue']:.4f} | {row['max_queue']:.4f} | "
                f"{row['mean_speed_kmh']:.4f} | {row['mean_reward']:.4f} | {row['switch_count']} | "
                f"{row['post_event_first_300s_mean_queue']:.4f} | {row['first_switch_after_event_s']} |"
            )

    lines.extend(
        [
            "",
            "## 结论摘要",
            "",
            "- 当前西侧强簇的综合最优策略是 `DQN-no-pred-west-5000`。",
            "- `DQN-pred-v1-west-2000` 在事故场景中表现出更早的事件后切相，但三场景综合排队仍高于 no-pred。",
            "- 不论 no-pred 还是 pred-v1，这组西侧路口都显著优于独立 MaxPressure，说明它是真正的联动强簇。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8-sig")


def plot_grouped_metrics(summary_rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = [
        ("mean_queue", "平均排队(veh)"),
        ("mean_speed_kmh", "平均速度(km/h)"),
        ("switch_count", "切换次数"),
    ]
    x = np.arange(3)
    width = 0.23
    for ax, (metric, title) in zip(axes, metrics):
        for idx, policy in enumerate(POLICIES):
            values = [
                next(r for r in summary_rows if r["scenario"] == scenario and r["policy"] == policy)[metric]
                for scenario in ["default", "incident", "control"]
            ]
            ax.bar(x + (idx - 1) * width, values, width=width, label=policy, color=COLORS[idx])
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIOS[s] for s in ["default", "incident", "control"]])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.6, 1.2), ncol=3, frameon=False)
    fig.suptitle("西侧 3 路口联动控制正式对比")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "policy_comparison_west_v1_formal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_queue_trajectories(loaded: dict[str, dict[str, list[dict[str, str]]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.5), sharex=False)
    for ax, scenario in zip(axes, ["default", "incident", "control"]):
        for policy, color in zip(POLICIES, COLORS):
            rows = loaded[policy][scenario]
            times = [to_float(r["sim_time_s"]) for r in rows]
            queues = [to_float(r["queue_sum"]) for r in rows]
            ax.plot(times, queues, label=policy, color=color, linewidth=1.8 if policy != "MaxPressure" else 1.5)
        if scenario in {"incident", "control"}:
            ax.axvline(1200, color="#d62728", linestyle="--", linewidth=1.2, label="事件开始")
        ax.set_title(f"{SCENARIOS[scenario]}：排队时序")
        ax.set_ylabel("queue_sum")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("仿真时间(s)")
    axes[0].legend(loc="upper right", ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "queue_trajectories_west_v1_formal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_evolution() -> None:
    scan_rows = load_rows(SCAN_DIR / "checkpoint_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in scan_rows:
        grouped.setdefault(row["model"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    titles = {
        "west_multi_no_pred_v1": "west no-pred checkpoint 演化",
        "west_multi_pred_v1": "west pred-v1 checkpoint 演化",
    }
    chosen = {
        "west_multi_no_pred_v1": 5000,
        "west_multi_pred_v1": 2000,
    }
    scenario_colors = {
        "default_mean_queue": "#00c2ff",
        "incident_mean_queue": "#ff5d73",
        "control_mean_queue": "#8ccf4d",
    }
    for ax, model_name in zip(axes, ["west_multi_no_pred_v1", "west_multi_pred_v1"]):
        rows = sorted(grouped[model_name], key=lambda item: int(item["checkpoint_step"]))
        steps = [int(row["checkpoint_step"]) for row in rows]
        for metric_key, label in [
            ("default_mean_queue", "默认运行态"),
            ("incident_mean_queue", "事故场景"),
            ("control_mean_queue", "配时扰动场景"),
        ]:
            values = [float(row[metric_key]) for row in rows]
            ax.plot(steps, values, marker="o", label=label, color=scenario_colors[metric_key])
        ax.axvline(chosen[model_name], color="#222", linestyle="--", linewidth=1.3, label=f"选定 {chosen[model_name]} 步")
        ax.set_title(titles[model_name])
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean_queue")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "checkpoint_evolution_west_v1_formal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_font()
    summary_rows: list[dict[str, float | str]] = []
    loaded: dict[str, dict[str, list[dict[str, str]]]] = {}
    for policy, scenario_map in SOURCES.items():
        loaded[policy] = {}
        for scenario, path in scenario_map.items():
            rows = load_rows(path)
            loaded[policy][scenario] = rows
            stats = summarize(rows, scenario)
            summary_rows.append(
                {
                    "policy": policy,
                    "scenario": scenario,
                    "scenario_label": SCENARIOS[scenario],
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
                }
            )

    write_table(summary_rows)
    plot_grouped_metrics(summary_rows)
    plot_queue_trajectories(loaded)
    plot_checkpoint_evolution()
    print("regenerated west_v1 formal artifacts")


if __name__ == "__main__":
    main()
