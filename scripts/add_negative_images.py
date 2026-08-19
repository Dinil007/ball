import os
import shutil
from pathlib import Path


def add_negative_images(
    src_dir="data/negative_images",
    dest_img_dir="data/ball_data/train/images",
    dest_lbl_dir="data/ball_data/train/labels"
):
    src_path = Path(src_dir)
    dest_img_path = Path(dest_img_dir)
    dest_lbl_path = Path(dest_lbl_dir)

    # 1. Check source folder exists
    if not src_path.exists() or not src_path.is_dir():
        print(f"ERROR: Source directory '{src_dir}' does not exist.")
        return

    # 2. Check destination folder exists (or create safely)
    dest_img_path.mkdir(parents=True, exist_ok=True)
    dest_lbl_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png"}

    initial_img_count = len([f for f in dest_img_path.iterdir() if f.suffix.lower() in valid_extensions])
    initial_lbl_count = len(list(dest_lbl_path.glob("*.txt")))

    print(f"Initial training images: {initial_img_count}")
    print(f"Initial training labels: {initial_lbl_count}")

    # 3. Copy image files avoiding duplicates
    copied_count = 0
    skipped_count = 0

    for file_path in src_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            dest_file = dest_img_path / file_path.name

            if dest_file.exists():
                print(f"Skipping duplicate: {file_path.name}")
                skipped_count += 1
            else:
                shutil.copy2(file_path, dest_file)
                copied_count += 1

    # 4. Verification checks
    final_img_count = len([f for f in dest_img_path.iterdir() if f.suffix.lower() in valid_extensions])
    final_lbl_count = len(list(dest_lbl_path.glob("*.txt")))

    # Verify no corresponding .txt files created for negative images
    negative_labels = [
        f.name for f in src_path.iterdir()
        if f.suffix.lower() in valid_extensions and (dest_lbl_path / f"{f.stem}.txt").exists()
    ]

    print("\n--- Summary ---")
    print(f"Negative images added: {copied_count}")
    print(f"Duplicates skipped: {skipped_count}")
    print(f"Final training images count: {final_img_count} (increased by {copied_count})")
    print(f"Final training labels count: {final_lbl_count} (unchanged)")

    if len(negative_labels) == 0:
        print("Verification PASSED: Negative images correctly have NO label (.txt) files.")
    else:
        print(f"WARNING: Found unexpected label files for negative images: {negative_labels}")


if __name__ == "__main__":
    add_negative_images()
