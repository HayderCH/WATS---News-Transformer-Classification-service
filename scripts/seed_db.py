"""Command line utility to seed the feedback database with demo content."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.db.seed import seed_initial_data  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing feedback/review items before inserting samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as session:
        result = seed_initial_data(session, overwrite=args.overwrite)

    inserted = result.as_dict()
    print(
        "Seed complete:",
        f"feedback={inserted['feedback']}",
        f"review_items={inserted['review_items']}",
        (
            "(use --overwrite to refresh)"
            if not args.overwrite
            else "(overwritten)"
        ),
    )


if __name__ == "__main__":
    main()
