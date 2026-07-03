from pathlib import Path

ROOT = Path("datasets/raw/NEU-DET/train/images")

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

total = 0

for folder in sorted(ROOT.iterdir()):

    if folder.is_dir():

        count = len(list(folder.glob("*")))

        total += count

        print(f"{folder.name:<20}{count}")

print("-" * 60)

print(f"Total Images : {total}")