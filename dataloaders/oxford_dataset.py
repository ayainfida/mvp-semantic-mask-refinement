from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF


class PetsDataset(Dataset):
    def __init__(self, root, train, image_transform, map_transform):
        prefix = root + ("/train_" if train else "/test_")
        images_dir = prefix + "images"
        self.images = torchvision.datasets.ImageFolder(images_dir, transform=image_transform)
        maps_dir = prefix + "maps"
        self.maps = torchvision.datasets.ImageFolder(maps_dir, transform=map_transform)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index][0]
        map = self.maps[index][0]
        return (image, map)

MEAN_RGB = [0.48501667380332947, 0.44971948862075806, 0.3968391418457031]
STDDEV_RGB = [0.26309531927108765, 0.25847315788269043, 0.2651668190956116]

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN_RGB, STDDEV_RGB)
])

def process_map(mapdata):
    # RGB to Grayscale and conversion to the integer labels 0, 1, 2.
    return (mapdata[0:1, :, :] * 255 / 100).long()


map_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    process_map
])