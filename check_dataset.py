from PIL import Image
import matplotlib.pyplot as plt

# Xem confusion matrix
img = Image.open("runs/detect/finetuned/train/confusion_matrix.png")
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()
