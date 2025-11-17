import os

# Base dataset path
BASE_DIR = r"D:\SBC_PROIECT\road_detection"
SETS = ["train", "test", "valid"]

def verify_dataset():
    for split in SETS:
        image_dir = os.path.join(BASE_DIR, split, "images")
        label_dir = os.path.join(BASE_DIR, split, "labels")

        # Get all filenames (without extension)
        image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith('.txt')}

        # Check mismatches
        missing_labels = image_files - label_files
        missing_images = label_files - image_files

        print(f"\n=== {split.upper()} SET ===")
        print(f"Total images: {len(image_files)} | Total labels: {len(label_files)}")

        if not missing_labels and not missing_images:
            print("✅ All images have matching labels.")
        else:
            if missing_labels:
                print(f"⚠️ Missing label files for {len(missing_labels)} images:")
                for name in sorted(missing_labels):
                    print(f"  {name}")
            if missing_images:
                print(f"⚠️ Missing image files for {len(missing_images)} labels:")
                for name in sorted(missing_images):
                    print(f"  {name}")

if __name__ == "__main__":
    verify_dataset()