from PIL import Image
import numpy as np
import torch
import os
import config
import torchvision.transforms.functional as TF
from torchvision import transforms 


def to_uint16(image):
    # Asegúrate de que esta función convierte tu imagen al formato uint16 correctamente
    return image.astype(np.uint16)

def single_channel_to_rgb(image):
    # Convierte una imagen de un solo canal a RGB replicando el canal
    return np.stack((image,)*3, axis=-1)

def preprocess_single_image(image_path):
    input_image = np.array(Image.open(image_path))
    # Convertir input_image a 3 canales si es en escala de grises
    if len(input_image.shape) == 2:  # Imagen en escala de grises
        input_image = np.stack((input_image,) * 3, axis=-1)
    elif input_image.shape[2] == 1:  # Imagen con un solo canal
        input_image = np.repeat(input_image, 3, axis=2)

    input_image = (input_image / 65535.0 * 255).astype(np.uint8)

    # Auxiliar target image
    target_image = np.zeros_like(input_image)

    augmentations = config.both_transform(image=input_image, image0=target_image)
    input_image = augmentations["image"]

    input_image = config.transform_only_input(image=input_image)["image"]

    return input_image



class MapDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.ids = [file.split('_')[0] for file in os.listdir(root_dir) if 'h.png' in file]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        heightmap_path = os.path.join(self.root_dir, f"{img_id}_h.png")
        texture_path = os.path.join(self.root_dir, f"{img_id}_t.png")

        input_image = np.array(Image.open(heightmap_path))
        target_image = np.array(Image.open(texture_path))

        # Convertir input_image a 3 canales si es en escala de grises
        if len(input_image.shape) == 2:  # Imagen en escala de grises
            input_image = np.stack((input_image,) * 3, axis=-1)
        elif input_image.shape[2] == 1:  # Imagen con un solo canal
            input_image = np.repeat(input_image, 3, axis=2)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_image = (input_image / 65535.0 * 255).astype(np.uint8)

        if input_image is None or target_image is None:
            raise ValueError(f"Error loading images: {heightmap_path} or {texture_path}")


        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]


        return input_image, target_image
    

def test():
    # Path to parent directory containing archive folder
    root_dir = "archive"

    dataset = MapDataset(root_dir=root_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        break


if __name__ == "__main__":
    test()