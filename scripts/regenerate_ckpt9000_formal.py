from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = ROOT / "reports" / "rl_signal_control" / "anticipatory_v3_checkpoint_scan"
OUT_DIR = ROOT / "reports" / "rl_signal_control" / "anticipatory_v3_checkpoint_9000_formal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "default": "\u9ed8\u8ba4\u8fd0\u884c\u6001",
    "incident": "\u4e8b\u6545\u573a\u666f",
    "control": "\u914d\u65f6\u6270\u52a8\u573a\u666f",
}

SOURCES = {
    "MaxPressure": {
        "default": SCAN_DIR / "default_max_pressure_eval.csv",
        "incident": SCAN_DIR / "incident_max_pressure_eval.csv",
        "control": SCAN_DIR / "control_max_pressure_eval.csv",
    },
    "DQN-pred-v1-v3-9000": {
        "default": SCAN_DIR / "default_ckpt_9000_eval.csv",
        "incident": SCAN_DIR / "incident_ckpt_9000_eval.csv",
        "control": SCAN_DIR / "control_ckpt_9000_eval.csv",
    },
    "DQN-pred-v1-v3-10000": {
        "default": SCAN_DIR / "default_ckpt_10000_eval.csv",
        "incident": SCAN_DIR / "incident_ckpt_10000_eval.csv",
        "control": SCAN_DIR / "control_ckpt_10000_eval.csv",
    },
}

POLICIES = ["MaxPressure", "DQN-pred-v1-v3-9000", "DQN-pred-v1-v3-10000"]
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
    switches = [int(float(r["switch_applied"])) for r in rows]
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

    if scenario == "incident":
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
    csv_path = OUT_DIR / "policy_comparison_ckpt9000_formal.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = OUT_DIR / "policy_comparison_ckpt9000_formal.md"
    lines = [
        "# 9000\u6b65 checkpoint \u6b63\u5f0f\u5bf9\u6bd4",
        "",
        "## \u6307\u6807\u8868",
        "",
        "| \u573a\u666f | \u7b56\u7565 | mean_queue | max_queue | mean_speed_kmh | mean_reward | switch_count | post_event_first_300s_mean_queue | first_switch_after_event_s |",
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
            "## \u7ed3\u8bba\u6458\u8981",
            "",
            "- `9000\u6b65` \u662f\u5f53\u524d\u4e09\u573a\u666f\u7efc\u5408\u6700\u5e73\u8861\u7684 checkpoint\u3002",
            "- \u76f8\u6bd4 `10000\u6b65`\uff0c`9000\u6b65` \u5728\u4e8b\u6545\u573a\u666f\u5e73\u5747\u6392\u961f\u548c\u4e8b\u4ef6\u540e300\u79d2\u6392\u961f\u66f4\u4f4e\u3002",
            "- \u76f8\u6bd4 `MaxPressure`\uff0c`9000\u6b65` \u5728\u9ed8\u8ba4\u573a\u666f\u548c\u4e8b\u6545\u573a\u666f\u6574\u4f53\u5e73\u5747\u6392\u961f\u66f4\u4f4e\uff0c"
            "\u4f46\u5728\u63a7\u5236\u6270\u52a8\u573a\u666f\u7565\u5dee\u3002",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8-sig")


def plot_grouped_metrics(summary_rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    metrics = [
        ("mean_queue", "\u5e73\u5747\u6392\u961f(veh)"),
        ("mean_speed_kmh", "\u5e73\u5747\u901f\u5ea6(km/h)"),
        ("switch_count", "\u5207\u6362\u6b21\u6570"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(3)
        width = 0.23
        for i, policy in enumerate(POLICIES):
            values = [
                next(r for r in summary_rows if r["scenario"] == scenario and r["policy"] == policy)[metric]
                for scenario in ["default", "incident", "control"]
            ]
            ax.bar(x + (i - 1) * width, values, width=width, label=policy, color=COLORS[i])
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIOS[s] for s in ["default", "incident", "control"]])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.6, 1.2), ncol=3, frameon=False)
    fig.suptitle("9000\u6b65 checkpoint \u6b63\u5f0f\u5bf9\u6bd4")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "policy_comparison_ckpt9000_formal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_queue_trajectories(loaded: dict[str, dict[str, list[dict[str, str]]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.5), sharex=False)
    for ax, scenario in zip(axes, ["default", "incident", "control"]):
        for policy, color in zip(POLICIES, COLORS):
            rows = loaded[policy][scenario]
            t = [to_float(r["sim_time_s"]) for r in rows]
            q = [to_float(r["queue_sum"]) for r in rows]
            ax.plot(t, q, label=policy, color=color, linewidth=1.8 if policy != "MaxPressure" else 1.5)
        if scenario == "incident":
            ax.axvline(1200, color="#d62728", linestyle="--", linewidth=1.2, label="\u4e8b\u4ef6\u5f00\u59cb")
        ax.set_title(f"{SCENARIOS[scenario]}\uff1a\u6392\u961f\u65f6\u5e8f")
        ax.set_ylabel("queue_sum")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("\u4eff\u771f\u65f6\u95f4(s)")
    axes[0].legend(loc="upper right", ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "queue_trajectories_ckpt9000_formal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_evolution() -> None:
    summary_rows = load_rows(SCAN_DIR / "checkpoint_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    step_values = [int(r["checkpoint_step"]) for r in summary_rows]
    for metric_key, label, color in [
        ("default_mean_queue", "\u9ed8\u8ba4\u8fd0\u884c\u6001", "#00c2ff"),
        ("incident_mean_queue", "\u4e8b\u6545\u573a\u666f", "#ff5d73"),
        ("control_mean_queue", "\u914d\u65f6\u6270\u52a8\u573a\u666f", "#8ccf4d"),
    ]:
        values = [float(r[metric_key]) for r in summary_rows]
        ax.plot(step_values, values, marker="o", label=label, color=color)
    ax.axvline(9000, color="#222", linestyle="--", linewidth=1.3, label="\u9009\u5b9a9000\u6b65")
    ax.set_title("anticipatory_v3 \u5404 checkpoint \u7684\u4e09\u573a\u666f\u5e73\u5747\u6392\u961f\u53d8\u5316")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean_queue")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "checkpoint_evolution_ckpt9000_formal.png", dpi=180, bbox_inches="tight")
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
    print("regenerated ckpt9000 formal artifacts")


if __name__ == "__main__":
    main()
