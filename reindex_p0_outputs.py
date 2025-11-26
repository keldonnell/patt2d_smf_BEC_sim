import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


FILE_PATTERN = re.compile(r"^(?P<prefix>psi|s)(?P<index>\d+)_(?P<p0>[0-9eE\.\-]+)\.out$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reindex patt1d output files so that indices increase monotonically with p0."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of the patt1d output directory (e.g. ivan_diss_params).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned renames without modifying any files.",
    )
    return parser.parse_args()


def collect_files(output_dir: Path) -> List[Tuple[str, Path, str, str]]:
    """
    Collect psi and s files, returning tuples of (prefix, path, index_str, p0_str).
    """
    entries: List[Tuple[str, Path, str, str]] = []
    for entry in output_dir.glob("*.out"):
        match = FILE_PATTERN.match(entry.name)
        if not match:
            continue
        prefix = match.group("prefix")
        index_str = match.group("index")
        p0_str = match.group("p0")
        entries.append((prefix, entry, index_str, p0_str))
    if not entries:
        raise FileNotFoundError(f"No psi/s files found in {output_dir}")
    return entries


def group_by_prefix(
    entries: List[Tuple[str, Path, str, str]]
) -> Dict[str, List[Tuple[Path, str, str]]]:
    grouped: Dict[str, List[Tuple[Path, str, str]]] = {"psi": [], "s": []}
    for prefix, path, index_str, p0_str in entries:
        grouped.setdefault(prefix, []).append((path, index_str, p0_str))
    # Ensure both psi and s exist
    if not grouped["psi"]:
        raise FileNotFoundError("No psi files detected; aborting.")
    if not grouped["s"]:
        raise FileNotFoundError("No s files detected; aborting.")
    return grouped


def validate_matching_pairs(grouped_entries: Dict[str, List[Tuple[Path, str, str]]]) -> None:
    psi_items = grouped_entries["psi"]
    s_items = grouped_entries["s"]
    if len(psi_items) != len(s_items):
        raise ValueError(
            f"Mismatched file counts: {len(psi_items)} psi files vs {len(s_items)} s files."
        )

    def signature(items: List[Tuple[Path, str, str]]) -> List[Tuple[float, str, str]]:
        sig: List[Tuple[float, str, str]] = []
        for _, index_str, p0_str in items:
            try:
                p0_val = float(p0_str)
            except ValueError:
                p0_val = float("inf")
            sig.append((p0_val, index_str, p0_str))
        sig.sort()
        return sig

    if signature(psi_items) != signature(s_items):
        raise ValueError(
            "Mismatch between psi and s file p0 values; manual inspection required before reindexing."
        )


def sort_paths_by_p0(
    paths: List[Tuple[Path, str, str]]
) -> List[Tuple[Path, str, str]]:
    def sort_key(item: Tuple[Path, str, str]):
        path, index_str, p0_str = item
        # Use float(p0) for ordering; fallback to string for stability if conversion fails.
        try:
            p0_val = float(p0_str)
        except ValueError:
            p0_val = float("inf")
        try:
            index_val = int(index_str)
        except ValueError:
            index_val = 0
        return (p0_val, index_val, path.name)

    return sorted(paths, key=sort_key)


def build_rename_plan(
    grouped_entries: Dict[str, List[Tuple[Path, str, str]]]
) -> List[Tuple[Path, Path]]:
    rename_plan: List[Tuple[Path, Path]] = []
    for prefix, items in grouped_entries.items():
        sorted_items = sort_paths_by_p0(items)
        for new_index, (path, index_str, p0_str) in enumerate(sorted_items):
            new_name = f"{prefix}{new_index}_{p0_str}.out"
            new_path = path.with_name(new_name)
            if path == new_path:
                continue
            rename_plan.append((path, new_path))
    return rename_plan


def perform_renames(rename_plan: List[Tuple[Path, Path]], dry_run: bool) -> None:
    if not rename_plan:
        print("All indices already sequential; nothing to rename.")
        return

    if dry_run:
        print("Planned renames:")
        for old_path, new_path in rename_plan:
            print(f"{old_path.name} -> {new_path.name}")
        return

    temp_suffix = ".reindex_tmp"
    temp_paths: List[Tuple[Path, Path]] = []
    try:
        # Stage 1: move to temporary names to avoid collisions
        for old_path, _ in rename_plan:
            temp_path = old_path.with_name(old_path.name + temp_suffix)
            os.rename(old_path, temp_path)
            temp_paths.append((old_path, temp_path))

        # Stage 2: rename from temporary to final names
        for old_path, new_path in rename_plan:
            temp_path = next(tp for op, tp in temp_paths if op == old_path)
            os.rename(temp_path, new_path)

    except Exception as exc:
        print(f"Error during renaming: {exc}")
        # Attempt to roll back staged files
        for original_path, temp_path in temp_paths:
            if temp_path.exists() and not original_path.exists():
                os.rename(temp_path, original_path)
        raise

    print(f"Renamed {len(rename_plan)} files.")


def main() -> None:
    args = parse_arguments()

    base_dir = Path("outputs")
    output_dir = base_dir / args.filename
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory '{output_dir}' not found.")

    entries = collect_files(output_dir)
    grouped_entries = group_by_prefix(entries)
    validate_matching_pairs(grouped_entries)
    rename_plan = build_rename_plan(grouped_entries)

    perform_renames(rename_plan, args.dry_run)


if __name__ == "__main__":
    main()
