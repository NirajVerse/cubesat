import os
import shutil

src_root = "dataset_split/images"      # has train/val/test
dst_root = "data/ohid_ff"              # should match DATASET_PATH in config.py

fire_dir = os.path.join(dst_root, "fire")
non_fire_dir = os.path.join(dst_root, "non_fire")
os.makedirs(fire_dir, exist_ok=True)
os.makedirs(non_fire_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    split_dir = os.path.join(src_root, split)
    if not os.path.isdir(split_dir):
        continue

    for fname in os.listdir(split_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src = os.path.join(split_dir, fname)
        label_char = fname[-5]   # your notebook logic
        if label_char == "1":
            dst = os.path.join(fire_dir, fname)
        elif label_char == "0":
            dst = os.path.join(non_fire_dir, fname)
        else:
            continue

        # copy; use copy2 to preserve metadata
        shutil.copy2(src, dst)

# print("Done. Created:")
# print(f"- {fire_dir}")
# print(f"- {non_fire_dir}")