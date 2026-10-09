"""Create (or reset) the SQLite database from the JSON seed files.

Usage:
    python -m scripts.init_db           # create tables + seed if empty
    python -m scripts.init_db --reset   # wipe and re-seed (demo reset)
"""

from __future__ import annotations

import argparse

from it_assistant.config import get_settings
from it_assistant.data.database import Database
from it_assistant.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="drop all data and re-seed")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)
    db = Database(settings.database_path)
    if args.reset:
        db.reset(settings.seed_dir)
        if settings.checkpoint_db_path.exists():
            settings.checkpoint_db_path.unlink()
        print(f"Database reset and re-seeded: {settings.database_path}")
    else:
        db.initialize(settings.seed_dir)
        print(f"Database ready: {settings.database_path}")


if __name__ == "__main__":
    main()
