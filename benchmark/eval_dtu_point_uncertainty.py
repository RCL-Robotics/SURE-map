#!/usr/bin/env python
"""Evaluate SURE-Map uncertainty filtering on DTU."""

from pathlib import Path

from eval_point_uncertainty import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "sure_map_dtu.yaml")
