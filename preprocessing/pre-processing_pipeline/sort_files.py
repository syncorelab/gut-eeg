from __future__ import annotations

import shutil
from pathlib import Path
import re


def copy_b3_c1ref_files(
    gut_tensors_dir: str | Path,
    processed_sub_withICA_dir: str | Path,
) -> None:
    """
    Copy each subject's Track_B3_c1ref.mat file from the gut_tensors folder
    into a shared output folder with the subject ID added to the filename.

    Expected source:
        gut_tensors/<subject_id>/dl_export/Track_B3_c1ref.mat

    Expected destination:
        processed_sub_withICA/<subject_id>_B3_c1ref.mat

    Only folders named with exactly 3 digits are treated as subject folders.
    Missing source files are skipped with a warning. If a destination file
    already exists, the script stops to avoid overwriting files.
    """

    gut_tensors_dir = Path(gut_tensors_dir)
    processed_sub_withICA_dir = Path(processed_sub_withICA_dir)

    if not gut_tensors_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {gut_tensors_dir}")

    if not gut_tensors_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {gut_tensors_dir}")

    if not processed_sub_withICA_dir.exists():
        raise FileNotFoundError(
            f"Destination folder does not exist: {processed_sub_withICA_dir}"
        )

    if not processed_sub_withICA_dir.is_dir():
        raise NotADirectoryError(
            f"Destination path is not a directory: {processed_sub_withICA_dir}"
        )

    # Match subject folders such as 001, 023, or 145.
    subject_pattern = re.compile(r"^\d{3}$")

    all_entries = sorted(gut_tensors_dir.iterdir(), key=lambda p: p.name)
    subject_dirs = [p for p in all_entries if p.is_dir() and subject_pattern.match(p.name)]

    if not subject_dirs:
        print(f"No 3-digit subject folders found in: {gut_tensors_dir}")
        return

    copied_count = 0
    skipped_count = 0
    warnings_list = []

    print(f"Source folder:      {gut_tensors_dir}")
    print(f"Destination folder: {processed_sub_withICA_dir}")
    print(f"Found {len(subject_dirs)} subject folders.\n")

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        source_file = subj_dir / "dl_export" / "Track_B3_c1ref.mat"
        dest_file = processed_sub_withICA_dir / f"{subj_id}_B3_c1ref.mat"

        if not source_file.exists():
            warning_msg = (
                f"[WARNING] Skipping subject {subj_id}: "
                f"missing file {source_file}"
            )
            print(warning_msg)
            warnings_list.append(warning_msg)
            skipped_count += 1
            continue

        if dest_file.exists():
            raise FileExistsError(
                f"Destination file already exists, stopping process:\n{dest_file}"
            )

        shutil.copy2(source_file, dest_file)
        print(f"[COPIED] {source_file} -> {dest_file}")
        copied_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subject folders checked: {len(subject_dirs)}")
    print(f"Files copied:            {copied_count}")
    print(f"Subjects skipped:        {skipped_count}")

    if warnings_list:
        print("\nWarnings:")
        for msg in warnings_list:
            print(msg)
    else:
        print("\nWarnings: None")


if __name__ == "__main__":
    # Set these paths to the input subject folders and the output folder.
    gut_tensors_dir = "/home/gutproject/Desktop/guteeg/gut-eeg/data/gut_tensors_ICA"
    processed_sub_withICA_dir = "/home/gutproject/Desktop/guteeg/gut-eeg/data/processed_sub_withICA_v2"

    copy_b3_c1ref_files(
        gut_tensors_dir=gut_tensors_dir,
        processed_sub_withICA_dir=processed_sub_withICA_dir,
    )