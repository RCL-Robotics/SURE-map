#!/usr/bin/env python3
"""Run end-to-end SURE-Map streaming pose estimation on VBR."""

from pathlib import Path

from _streaming_pose import cli_main


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    cli_main("vbr", repo / "online" / "configs" / "vbr.yaml")
