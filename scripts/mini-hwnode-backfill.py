#!/usr/bin/env python3
# mini-hwnode-backfill.py
# Fix hardware_node: "archon" → "mini" in all mini-* batch JSON files.
# Scope: ~/Archon-Vault/Projects/divergence-router/data/mini-*/v2.1/*.json
# Safe: writes to .tmp then renames. Idempotent.
# Usage: python3 mini-hwnode-backfill.py [--dry-run]

import json
import os
import sys
import tempfile
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv
DATA_ROOT = Path.home() / "Archon-Vault/Projects/divergence-router/data"

WRONG = "archon"
RIGHT = "mini"


def fix_value(obj):
    """Recursively replace hardware_node: 'archon' with 'mini'. Returns (new_obj, change_count)."""
    changes = 0
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "hardware_node" and v == WRONG:
                new_obj[k] = RIGHT
                changes += 1
            else:
                fixed_v, sub_changes = fix_value(v)
                new_obj[k] = fixed_v
                changes += sub_changes
        return new_obj, changes
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            fixed_item, sub_changes = fix_value(item)
            new_list.append(fixed_item)
            changes += sub_changes
        return new_list, changes
    else:
        return obj, 0


def process_file(path: Path):
    """Returns (was_modified, fields_changed)."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as e:
        print(f"  SKIP {path}: {e}")
        return False, 0

    new_data, changes = fix_value(data)
    if changes == 0:
        return False, 0

    if not DRY_RUN:
        # Safe write: temp file in same dir, then atomic rename
        dir_ = path.parent
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dir_, suffix=".tmp", delete=False
        ) as tf:
            json.dump(new_data, tf, indent=2, ensure_ascii=False)
            tmp_path = tf.name
        os.replace(tmp_path, path)

    return True, changes


def main():
    mode_label = "DRY-RUN" if DRY_RUN else "LIVE"
    print(f"mini-hwnode-backfill — {mode_label}")
    print(f"Scanning: {DATA_ROOT}/mini-*/v2.1/*.json\n")

    glob_pattern = "mini-*/v2.1/*.json"
    all_files = sorted(DATA_ROOT.glob(glob_pattern))

    scanned = 0
    modified = 0
    total_fields = 0

    for path in all_files:
        scanned += 1
        was_modified, fields = process_file(path)
        if was_modified:
            modified += 1
            total_fields += fields

    print(f"Files scanned:   {scanned:,}")
    print(f"Files modified:  {modified:,}")
    print(f"Fields changed:  {total_fields:,}")
    if DRY_RUN:
        print("\nDry run — no files written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
