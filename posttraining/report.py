"""Locate an existing user reply gallery without reading or scoring replies."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    gallery = args.run_dir / "replies.html"
    if not gallery.is_file():
        raise SystemExit("No new-format reply gallery exists here. Historical artifacts are preserved; automated review is disabled.")
    print(f"User review only: {gallery.resolve()}")


if __name__ == "__main__":
    main()
