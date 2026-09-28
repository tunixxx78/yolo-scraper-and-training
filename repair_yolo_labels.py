#!/usr/bin/env python3
"""
repair_yolo_labels.py

Repairs YOLO .txt label files where literal '\\n' sequences were accidentally
written instead of real newline characters.

Default behavior:
- recursively scans <dataset>/train/labels, val/labels and test/labels
- validates every non-empty YOLO detection row
- creates a .bak backup before changing a file
- does not modify already-correct files

Usage:
    python repair_yolo_labels.py datasets/rovajarvi

Dry run:
    python repair_yolo_labels.py datasets/rovajarvi --dry-run

Skip backups:
    python repair_yolo_labels.py datasets/rovajarvi --no-backup
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def validate_yolo_text(text: str, path: Path) -> list[str]:
    """Validate standard YOLO detection rows and return normalized rows."""
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    normalized: list[str] = []

    for line_no, row in enumerate(rows, start=1):
        parts = row.split()
        if len(parts) != 5:
            raise ValueError(
                f"{path}: line {line_no}: expected 5 values, got {len(parts)}: {row!r}"
            )

        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{path}: line {line_no}: invalid numeric value: {row!r}"
            ) from exc

        if class_id < 0:
            raise ValueError(f"{path}: line {line_no}: negative class id {class_id}")

        if any(v < 0.0 or v > 1.0 for v in coords):
            raise ValueError(
                f"{path}: line {line_no}: coordinates must be in [0, 1]: {row!r}"
            )

        normalized.append(
            f"{class_id} " + " ".join(f"{v:.6f}" for v in coords)
        )

    return normalized


def repair_file(path: Path, dry_run: bool, backup: bool) -> tuple[str, str]:
    original = path.read_text(encoding="utf-8")

    # The actual bug: files contain the two literal characters "\" and "n".
    candidate = original.replace("\\n", "\n")

    try:
        rows = validate_yolo_text(candidate, path)
    except ValueError as exc:
        return "invalid", str(exc)

    # Empty files are valid negative examples.
    repaired = ("\n".join(rows) + "\n") if rows else ""

    if repaired == original:
        return "ok", ""

    if not dry_run:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
        path.write_text(repaired, encoding="utf-8")

    return "fixed", ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair YOLO labels containing literal \\\\n sequences."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset root, e.g. datasets/rovajarvi",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .txt.bak backups.",
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        print(f"ERROR: dataset does not exist: {dataset}")
        return 2

    label_dirs = [
        dataset / split / "labels"
        for split in ("train", "val", "test")
        if (dataset / split / "labels").is_dir()
    ]

    if not label_dirs:
        print(f"ERROR: no train/labels, val/labels or test/labels found under: {dataset}")
        return 2

    files: list[Path] = []
    for label_dir in label_dirs:
        files.extend(sorted(label_dir.rglob("*.txt")))

    ok = fixed = invalid = 0
    errors: list[str] = []

    for path in files:
        status, message = repair_file(
            path,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )

        if status == "ok":
            ok += 1
        elif status == "fixed":
            fixed += 1
            action = "WOULD FIX" if args.dry_run else "FIXED"
            print(f"{action}: {path}")
        else:
            invalid += 1
            errors.append(message)
            print(f"INVALID: {message}")

    print()
    print("Summary")
    print(f"  Dataset:       {dataset}")
    print(f"  Label files:   {len(files)}")
    print(f"  Already OK:    {ok}")
    print(f"  {'Would fix' if args.dry_run else 'Fixed'}:        {fixed}")
    print(f"  Invalid left:  {invalid}")
    print(f"  Backups:       {'disabled' if args.no_backup else 'enabled (.txt.bak)'}")
    print(f"  Mode:          {'DRY RUN' if args.dry_run else 'WRITE'}")

    if errors:
        print()
        print("Some files still need manual inspection.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
