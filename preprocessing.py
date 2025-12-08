import os
import shutil
import random
from pathlib import Path

# === CẤU HÌNH ===
KNIFE_DIR = "./dataset/knife.yolov11"
SCISSORS_DIR = "./dataset/scissors.yolov11"
OUTPUT_DIR = "./dataset/merge.yolov11"

TRAIN_RATIO = 0.75
VALID_RATIO = 0.15
TEST_RATIO = 0.10

random.seed(42)

# === TẠO THƯ MỤC ===
for split in ["train", "valid", "test"]:
    Path(f"{OUTPUT_DIR}/{split}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/{split}/labels").mkdir(parents=True, exist_ok=True)


def collect_all_samples(src_dir, prefix, class_mapping):
    """Thu thập tất cả samples từ train/valid/test"""
    samples = []
    for split in ["train", "valid", "test"]:
        img_dir = f"{src_dir}/{split}/images"
        lbl_dir = f"{src_dir}/{split}/labels"

        if not os.path.exists(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            samples.append(
                {
                    "img_src": f"{img_dir}/{img_file}",
                    "lbl_src": f"{lbl_dir}/{lbl_file}",
                    "img_name": f"{prefix}_{img_file}",
                    "lbl_name": f"{prefix}_{lbl_file}",
                    "class_mapping": class_mapping,
                }
            )
    return samples


def process_and_copy(sample, split):
    """Copy image và remap label"""
    dst_img = f"{OUTPUT_DIR}/{split}/images/{sample['img_name']}"
    shutil.copy2(sample["img_src"], dst_img)

    src_lbl = sample["lbl_src"]
    dst_lbl = f"{OUTPUT_DIR}/{split}/labels/{sample['lbl_name']}"

    if os.path.exists(src_lbl):
        with open(src_lbl, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                old_class = int(parts[0])
                new_class = sample["class_mapping"].get(old_class, old_class)
                parts[0] = str(new_class)
                new_lines.append(" ".join(parts))
        with open(dst_lbl, "w") as f:
            f.write("\n".join(new_lines))


# === THU THẬP DATA ===
knife_samples = collect_all_samples(KNIFE_DIR, "knife", {0: 0})
scissors_samples = collect_all_samples(SCISSORS_DIR, "scissors", {0: 1, 1: 1})

print(f"Knife: {len(knife_samples)}")
print(f"Scissors: {len(scissors_samples)}")

# === CÂN BẰNG DATA (undersample) ===
min_count = min(len(knife_samples), len(scissors_samples))

random.shuffle(knife_samples)
random.shuffle(scissors_samples)

knife_samples = knife_samples[:min_count]
scissors_samples = scissors_samples[:min_count]

print(f"\nSau cân bằng: {min_count} mỗi class")

# === GỘP & SHUFFLE ===
all_samples = knife_samples + scissors_samples
random.shuffle(all_samples)

# === SPLIT 75/15/10 ===
n = len(all_samples)
train_end = int(n * TRAIN_RATIO)
valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

splits = {
    "train": all_samples[:train_end],
    "valid": all_samples[train_end:valid_end],
    "test": all_samples[valid_end:],
}

# === COPY FILES ===
for split, samples in splits.items():
    for sample in samples:
        process_and_copy(sample, split)

# === TẠO data.yaml ===
yaml_content = """train: ./train/images
val: ./valid/images
test: ./test/images

nc: 2
names: ['knife', 'scissors']
"""

with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

# === THỐNG KÊ ===
print("\n=== KẾT QUẢ ===")
print(f"Total: {n}")
print(f"Train: {len(splits['train'])} ({len(splits['train']) / n * 100:.1f}%)")
print(f"Valid: {len(splits['valid'])} ({len(splits['valid']) / n * 100:.1f}%)")
print(f"Test: {len(splits['test'])} ({len(splits['test']) / n * 100:.1f}%)")
print(f"\nOutput: {OUTPUT_DIR}")
