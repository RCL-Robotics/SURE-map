#!/usr/bin/env python3
"""Run end-to-end SURE-Map streaming pose estimation on Oxford Spires."""

from pathlib import Path

from _streaming_pose import cli_main


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    cli_main("oxford", repo / "online" / "configs" / "oxford.yaml")
