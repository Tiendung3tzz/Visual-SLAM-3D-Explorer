from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual SLAM Dashboard")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "visual_slam_upgrade",
        help="Notebook artifact folder used by Replay Mode",
    )
    return parser


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    from app.ui.main_window import MainWindow

    args = build_parser().parse_args()
    application = QApplication(sys.argv)
    window = MainWindow(args.artifacts)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

