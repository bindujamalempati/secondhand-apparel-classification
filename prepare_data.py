from __future__ import annotations

import argparse
from src.data_utils import download_and_prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a real-world apparel image subset.")
    parser.add_argument("--limit_per_class", type=int, default=250, help="Number of images to keep per class")
    args = parser.parse_args()

    df = download_and_prepare(limit_per_class=args.limit_per_class)
    print(f"Prepared {len(df)} processed images across {df['label'].nunique()} classes.")
    print(df.groupby(['split', 'label']).size())


if __name__ == "__main__":
    main()