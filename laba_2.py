import torch
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d

warnings.filterwarnings("ignore")

# Загрузка и анализ изображения
image_path = "butterfly.jpg"
original_image = Image.open(image_path)

print(f"Формат изображения: {original_image.format}")
print(f"Размеры: {original_image.size}")
print(f"Цветовая схема: {original_image.mode}")

# Преобразование в тензор
image_array = np.array(original_image)
print(f"Форма массива изображения: {image_array.shape}")

image_tensor = torch.tensor(image_array, dtype=torch.float32)
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
print(f"Форма тензора изображения: {image_tensor.shape}")

# Создание случайного ядра свертки
torch.manual_seed(42)
random_kernel = torch.randn(1, 3, 3, 3, dtype=torch.float32)  # [выходные_каналы, входные_каналы, высота, ширина]
print(f"Форма случайного ядра: {random_kernel.shape}")
print(f"Значения случайного ядра:\n{random_kernel}")

# Применение свертки
convolved_image = conv2d(image_tensor, random_kernel)
convolved_image = convolved_image.permute(0, 2, 3, 1)  # [1, H, W, 1]
print(f"Форма свернутого изображения: {convolved_image.shape}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image_array)
axes[0].set_title('Оригинальное изображение')
axes[0].axis('off')

# Используем абсолютные значения для лучшей визуализации
axes[1].imshow(torch.abs(convolved_image[0, :, :, 0]), cmap='gray')
axes[1].set_title('Свертка со случайным ядром (абсолютные значения)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Детальная визуализация свернутого изображения
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(convolved_image[0, :, :, 0].numpy(), cmap='RdYlBu')
plt.colorbar(heatmap, fraction=0.046, pad=0.04)
plt.title('Свертка со случайным ядром (исходные значения)')
plt.axis('off')
plt.tight_layout()
plt.show()

# Визуализация ядер свертки
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
channel_names = ['Красный', 'Зеленый', 'Синий']

for i in range(3):
    kernel_channel = random_kernel[0, i, :, :]
    im = axes[i].imshow(kernel_channel, cmap='RdYlBu', vmin=-2, vmax=2)
    axes[i].set_title(f'Канал {i+1}: {channel_names[i]}')
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle('Ядра свертки по цветовым каналам', y=1.05)
plt.tight_layout()
plt.show()