import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF


class COCODataset(Dataset):
    """
    COCO pets dataset:
      - root/images: RGB .jpg
      - root/masks:  0 = bg, 1 = cat, 2 = dog (uint8 .png)
    """

    def __init__(self, root, train, image_transform, map_transform,
                 train_split=0.85, shuffle=True):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "masks")
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.train = train

        # list all .jpg images
        all_images = sorted(
            [f for f in os.listdir(self.images_dir) if f.lower().endswith(".jpg")]
        )

        if shuffle:
            random.seed(42)
            random.shuffle(all_images)

        split_idx = int(len(all_images) * train_split)
        if train:
            self.images = all_images[:split_idx]
        else:
            self.images = all_images[split_idx:]

        print(f"{'Train' if train else 'Val'} set: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]

        # image
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # mask (same stem, .png), values in {0,1,2}
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path)

        # apply transforms
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.map_transform is not None:
            mask = self.map_transform(mask)

        return image, mask


# ==========================
# Transforms
# ==========================
MEAN_RGB = [0.46021568775177, 0.44226375222206116, 0.41804927587509155]
STDDEV_RGB = [0.2755761742591858, 0.26984524726867676, 0.2828117907047272]

coco_image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN_RGB, STDDEV_RGB),
])

def coco_process_map(mask_tensor):
    """
    mask_tensor: [1,H,W] uint8 from PILToTensor, values 0/1/2
    returns: [1,H,W] Long with same class indices
    """
    return mask_tensor.long()

coco_map_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(
        (256, 256),
        interpolation=TF.InterpolationMode.NEAREST
    ),
    torchvision.transforms.PILToTensor(),   # preserves integer labels
    coco_process_map
])