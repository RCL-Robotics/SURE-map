#!/usr/bin/env python
"""Evaluate SURE-Map uncertainty filtering on Neural RGB-D."""

from pathlib import Path

from eval_point_uncertainty import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "sure_map_neural_rgbd.yaml")
