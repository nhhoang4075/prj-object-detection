import os


def count_labels(label_dir):
    counts = {0: 0, 1: 0}
    for f in os.listdir(label_dir):
        with open(f"{label_dir}/{f}") as file:
            for line in file:
                cls = int(line.split()[0])
                counts[cls] += 1
    return counts


print("Train:", count_labels("./dataset/merge.yolov11/train/labels"))
print("Valid:", count_labels("./dataset/merge.yolov11/valid/labels"))
print("Test:", count_labels("./dataset/merge.yolov11/test/labels"))
