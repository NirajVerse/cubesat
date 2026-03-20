# %%
import os
import shutil
import random

# %%
# Paths
dataset_dir = "./YOLODataset/images"   # folder with all 1197 images
output_dir = "dataset_split"     # new folder for train/val/test

# %%
# Create directories
splits = ["train", "val", "test"]
classes = ["positive", "negative"]

# %%
# Create YOLO folders
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)

# %%
# Collect files by class (positive=1, negative=0 based on last char)
positive_files = [f for f in os.listdir(dataset_dir) if f.endswith(('jpg','png','jpeg')) and f[-5] == '1']
negative_files = [f for f in os.listdir(dataset_dir) if f.endswith(('jpg','png','jpeg')) and f[-5] == '0']

# %%
print("Positive:", len(positive_files))
print("Negative:", len(negative_files))
print("Total:", len(positive_files) + len(negative_files))

# %%
# Shuffle for randomness
random.shuffle(positive_files)
random.shuffle(negative_files)

# %%
# Function to split files
def split_files(file_list, train_ratio=0.4, val_ratio=0.1):
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train+n_val]
    test_files = file_list[n_train+n_val:]
    return train_files, val_files, test_files

# %%
# Split positive and negative sets
pos_train, pos_val, pos_test = split_files(positive_files)
neg_train, neg_val, neg_test = split_files(negative_files)


# %%
# Helper: copy image only (no labels)
def copy_images_only(files, split):
    for f in files:
        src = os.path.join(dataset_dir, f)
        dst_img = os.path.join(output_dir, "images", split, f)
        shutil.copy(src, dst_img)

# %%
# Copy only images (no label files)
copy_images_only(pos_train, "train")
copy_images_only(pos_val, "val")
copy_images_only(pos_test, "test")

copy_images_only(neg_train, "train")
copy_images_only(neg_val, "val")
copy_images_only(neg_test, "test")

# %%
# Path to your dataset split folder
dataset_dir = "dataset_split/images"  # adjust if needed

splits = ["train", "val", "test"]
for split in splits:
    split_path = os.path.join(dataset_dir, split)
    if os.path.exists(split_path):
        files = [f for f in os.listdir(split_path) if f.lower().endswith(('jpg','png','jpeg'))]
        
        total = len(files)
        positives = sum(1 for f in files if f[-5] == '1')  # last char before extension
        negatives = sum(1 for f in files if f[-5] == '0')
        
        print(f"\n{split.capitalize()} set:")
        print(f"  Total: {total}")
        print(f"  Positive: {positives}")
        print(f"  Negative: {negatives}")
    else:
        print(f"\n{split.capitalize()} set: Folder not found")

# %%


# %%
import pandas as pd

def create_file_list(dataset_dir="dataset_split/images"):
    """
    Create a list of files in train, val, test folders 
    with class labels (positive/negative).
    
    Args:
        dataset_dir (str): Path to dataset images folder.
        
    Returns:
        pd.DataFrame: DataFrame with columns [filename, split, class]
    """
    
    splits = ["train", "val", "test"]
    records = []

    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            files = [f for f in os.listdir(split_path) if f.lower().endswith(('jpg','png','jpeg'))]
            for f in files:
                label = "positive" if f[-5] == '1' else "negative"
                records.append([f, split, label])
        else:
            print(f"⚠️ Folder not found: {split_path}")

    df = pd.DataFrame(records, columns=["filename", "split", "class"])
    return df

# Example usage
df_files = create_file_list("dataset_split/images")
print(df_files.head())

# Optionally save to CSV
df_files.to_csv("file_list.csv", index=False)
print("✅ File list saved to file_list.csv")


