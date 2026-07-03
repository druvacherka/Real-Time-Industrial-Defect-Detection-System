from pathlib import Path

ROOT = Path("datasets/raw/NEU-DET")

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

assert ROOT.exists(), "Dataset directory not found!"

splits = ["train", "validation"]

for split in splits:

    print(f"\nChecking {split.upper()}")

    split_path = ROOT / split

    image_path = split_path / "images"
    annotation_path = split_path / "annotations"

    print(f"Images Folder      : {image_path.exists()}")
    print(f"Annotations Folder : {annotation_path.exists()}")

    image_count = len(list(image_path.rglob("*.*")))
    xml_count = len(list(annotation_path.glob("*.xml")))

    print(f"Total Images       : {image_count}")
    print(f"Total XML Files    : {xml_count}")

print("\nDataset verification completed successfully.")