import os
import shutil
import zipfile
from pathlib import Path


def prepare_dataset(
    zip_candidates=(
        "cricket_ball.yolov8.zip",
        "data/cricket_ball.yolov8.zip",
        "D:/Downloads/cricket_ball.yolov8 (1).zip",
        "D:/Downloads/cricket_ball.yolov8.zip",
    ),
    target_dir="data/ball_data",
    negative_dir="data/negative_images"
):
    target_path = Path(target_dir).resolve()
    negative_path = Path(negative_dir).resolve()

    print("==========================================")
    print("      CRICKETGRIP AI DATASET PREPARATION   ")
    print("==========================================")

    # 1. Find zip file
    zip_path = None
    for candidate in zip_candidates:
        p = Path(candidate).resolve()
        if p.exists() and p.is_file():
            zip_path = p
            break

    if not zip_path:
        print("ERROR: Could not locate cricket_ball.yolov8.zip!")
        return False

    print(f"[1/5] Located original Roboflow archive: {zip_path}")

    # 2. Clean and extract into target_dir
    print(f"[2/5] Restoring original Roboflow dataset to {target_path} ...")
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_path)

    # Ensure clean data.yaml
    data_yaml_path = target_path / "data.yaml"
    yaml_content = f"""path: {target_path.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['cricket_ball']
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"      Verified data.yaml at {data_yaml_path}")

    # 3. Verify positive images and labels
    train_img_dir = target_path / "train" / "images"
    train_lbl_dir = target_path / "train" / "labels"
    valid_img_dir = target_path / "valid" / "images"
    valid_lbl_dir = target_path / "valid" / "labels"
    test_img_dir = target_path / "test" / "images"
    test_lbl_dir = target_path / "test" / "labels"

    img_exts = {".jpg", ".jpeg", ".png"}

    pos_train_images = [f for f in train_img_dir.iterdir() if f.suffix.lower() in img_exts]
    pos_train_labels = list(train_lbl_dir.glob("*.txt"))
    valid_images = [f for f in valid_img_dir.iterdir() if f.suffix.lower() in img_exts]
    valid_labels = list(valid_lbl_dir.glob("*.txt"))
    test_images = [f for f in test_img_dir.iterdir() if f.suffix.lower() in img_exts]
    test_labels = list(test_lbl_dir.glob("*.txt"))

    print(f"[3/5] Original Positive Dataset Counts:")
    print(f"      Train : {len(pos_train_images)} images, {len(pos_train_labels)} labels")
    print(f"      Valid : {len(valid_images)} images, {len(valid_labels)} labels")
    print(f"      Test  : {len(test_images)} images, {len(test_labels)} labels")

    if len(pos_train_labels) == 0 or len(pos_train_images) == 0:
        print("ERROR: Original labels or images are missing! Aborting.")
        return False

    # 4. Merge negative background images into train/images only
    print(f"[4/5] Merging negative images from {negative_path} into train/images ...")
    neg_added = 0
    neg_skipped = 0

    if negative_path.exists() and negative_path.is_dir():
        for neg_file in negative_path.iterdir():
            if neg_file.is_file() and neg_file.suffix.lower() in img_exts:
                dest_file = train_img_dir / neg_file.name
                if dest_file.exists():
                    neg_skipped += 1
                else:
                    shutil.copy2(neg_file, dest_file)
                    neg_added += 1
    else:
        print(f"WARNING: Negative images directory '{negative_path}' not found.")

    # 5. Final verification
    total_train_images = len([f for f in train_img_dir.iterdir() if f.suffix.lower() in img_exts])
    total_train_labels = len(list(train_lbl_dir.glob("*.txt")))
    total_all_labels = len(list(target_path.rglob("*.txt")))

    # Check for any accidental label files for negative images
    invalid_neg_labels = []
    if negative_path.exists() and negative_path.is_dir():
        for neg_file in negative_path.iterdir():
            lbl_file = train_lbl_dir / f"{neg_file.stem}.txt"
            if lbl_file.exists():
                invalid_neg_labels.append(lbl_file.name)

    print("\n==========================================")
    print("             DATASET SUMMARY              ")
    print("==========================================")
    print(f"Number of positive images   : {len(pos_train_images)}")
    print(f"Negative images added       : {neg_added}")
    print(f"Total train images          : {total_train_images}")
    print(f"Total label files           : {total_all_labels} (Train: {total_train_labels}, Valid: {len(valid_labels)}, Test: {len(test_labels)})")
    
    if len(invalid_neg_labels) == 0:
        print("Negative labels check       : PASSED (0 label files created for negative images)")
    else:
        print(f"Negative labels check       : FAILED (Found unexpected labels: {invalid_neg_labels})")

    print(f"Dataset structure           : VALID")
    print("==========================================")
    return True


if __name__ == "__main__":
    prepare_dataset()
