from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KEEP_PATHS = [
    "outputs/pairs.csv",
    "outputs/weights.csv",
    "outputs/weights_template.csv",
    "outputs/pair_contact_sheet.jpg",
    "outputs/roi_config.json",
    "outputs/scale_config.json",
    "outputs/image_size_check.csv",
    "outputs/roi_tomato_images",
    "outputs/review_pairs",
    "outputs/review_contact_sheet.jpg",
    "outputs/debug_scale",
    "outputs/debug_masks_red",
    "outputs/volume_results_red",
    "outputs/debug_masks_contact_sheet_red.jpg",
    "outputs/summary_red.xlsx",
    "outputs/seg_dataset",
    "outputs/seg_dataset_preview",
]

ARCHIVE_PATHS = [
    "outputs/debug_masks",
    "outputs/volume_results",
    "outputs/debug_masks_contact_sheet.jpg",
    "outputs/summary.xlsx",
    "outputs/debug_masks_warm",
    "outputs/volume_results_warm",
    "outputs/debug_masks_contact_sheet_warm.jpg",
    "outputs/summary_warm.xlsx",
    "outputs/mask_mode_comparison.csv",
    "outputs/mask_mode_summary.csv",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def print_section(title: str, rows: list[str]) -> None:
    print(f"\n[{title}]")
    if not rows:
        print("  (none)")
        return
    for row in rows:
        print(f"  {row}")


def find_uncovered_outputs(covered: set[Path]) -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    uncovered: list[Path] = []
    for child in sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.name):
        if child not in covered:
            uncovered.append(child)
    return uncovered


def make_report_row(
    action: str,
    source: Path,
    destination: Path | None,
    exists: bool,
    status: str,
    note: str,
) -> dict[str, str]:
    return {
        "action": action,
        "source": str(source),
        "destination": "" if destination is None else str(destination),
        "exists": str(exists),
        "status": status,
        "note": note,
    }


def write_report(report_path: Path, rows: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["action", "source", "destination", "exists", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)


def run_cleanup(apply: bool) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = PROJECT_ROOT / f"outputs_archive_cleanup_{timestamp}"
    report_path = OUTPUTS_DIR / f"cleanup_report_{timestamp}.csv"

    keep_sources = [project_path(p) for p in KEEP_PATHS]
    archive_sources = [project_path(p) for p in ARCHIVE_PATHS]
    covered = set(keep_sources + archive_sources)
    uncovered = find_uncovered_outputs(covered)

    report_rows: list[dict[str, str]] = []
    keep_lines: list[str] = []
    archive_lines: list[str] = []
    missing_archive_lines: list[str] = []
    uncovered_lines: list[str] = []

    for source in keep_sources:
        exists = source.exists()
        status = "kept" if exists else "missing"
        note = "required keep path" if exists else "required keep path is missing"
        keep_lines.append(f"{source.relative_to(PROJECT_ROOT)} [{'exists' if exists else 'missing'}]")
        report_rows.append(make_report_row("keep", source, None, exists, status, note))

    for source in archive_sources:
        exists = source.exists()
        destination = archive_root / source.relative_to(PROJECT_ROOT)
        if exists:
            archive_lines.append(f"{source.relative_to(PROJECT_ROOT)} -> {destination.relative_to(PROJECT_ROOT)}")
        else:
            missing_archive_lines.append(str(source.relative_to(PROJECT_ROOT)))

        if not exists:
            report_rows.append(make_report_row("archive", source, destination, False, "missing", "expected archive path not found"))
            continue

        if not apply:
            report_rows.append(make_report_row("archive", source, destination, True, "dry-run", "would move with --apply"))
            continue

        if destination.exists():
            report_rows.append(make_report_row("archive", source, destination, True, "skipped", "destination already exists"))
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        report_rows.append(make_report_row("archive", source, destination, True, "moved", "moved to archive"))

    for source in uncovered:
        uncovered_lines.append(str(source.relative_to(PROJECT_ROOT)))
        report_rows.append(make_report_row("uncovered", source, None, source.exists(), "not_touched", "not covered by keep/archive rules"))

    write_report(report_path, report_rows)

    print(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    print(f"Archive directory: {archive_root}")
    print(f"Cleanup report: {report_path}")
    print_section("Keep", keep_lines)
    print_section("Archive", archive_lines)
    print_section("Expected archive paths not found", missing_archive_lines)
    print_section("Other outputs not covered by rules", uncovered_lines)
    if not apply:
        print("\nNo files were moved. Re-run with --apply after reviewing this dry-run.")
    else:
        print("\nArchive move finished. No files were deleted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run-first cleanup for project outputs.")
    parser.add_argument("--apply", action="store_true", help="Actually move archive targets. Default is dry-run only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cleanup(apply=args.apply)


if __name__ == "__main__":
    main()
