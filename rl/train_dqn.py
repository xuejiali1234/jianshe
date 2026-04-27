from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder DQN entrypoint for the next training phase.")
    parser.add_argument("--config", default="configs/rl_signal_config.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    print(
        "DQN training is intentionally not executed in this non-retraining phase. "
        f"Config is ready for the next phase: {config_path}"
    )


if __name__ == "__main__":
    main()
