import math
import random
import numpy as np
import blobfile as bf
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    Load spectral patches from either a directory (image files) or a `.npy` file.
    """
    if data_dir.endswith(".npy"):  # If input is an .npy file
        all_data = np.load(data_dir)  # Load the entire dataset
        dataset = NpyDataset(
            all_data,  # Pass the loaded array
            image_size=image_size,
            random_crop=random_crop,
            random_flip=random_flip,
        )
    else:  # Load images from a directory
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=random_crop,
            random_flip=random_flip,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Recursively list all image files in a directory.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class NpyDataset(Dataset):
    """
    Dataset class for handling .npy input files.
    """

    def __init__(self, data_array, image_size, random_crop=False, random_flip=True):
        super().__init__()
        self.data = data_array  # The entire dataset is loaded in memory
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = self.data[idx]  # Extract the individual image (already as NumPy array)

        if self.random_crop:
            arr = random_crop_arr(arr, self.image_size)
        else:
            arr = center_crop_arr(arr, self.image_size)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]  # Flip along width

        arr = arr.astype(np.float32)  # Ensure float32 format

        if len(arr.shape) == 2:  # Convert to (1, H, W) for grayscale compatibility
            arr = np.expand_dims(arr, axis=0)

        return arr, {}  # No class labels


class ImageDataset(Dataset):
    """
    Dataset class for handling image file inputs.
    """

    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths  # Image file paths
        self.local_classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            image = np.load(f)  # Load .npy file instead of image file

        if self.random_crop:
            image = random_crop_arr(image, self.resolution)
        else:
            image = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            image = image[:, ::-1]

        image = image.astype(np.float32)  # Convert to float32

        if len(image.shape) == 2:  # Convert to (1, H, W) for grayscale
            image = np.expand_dims(image, axis=0)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return image, out_dict


def center_crop_arr(arr, image_size):
    """
    Center crops a NumPy array (not a PIL image) to the specified image size.
    """
    h, w = arr.shape  # Ensure input is (H, W)
    crop_y = (h - image_size) // 2
    crop_x = (w - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(arr, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    Randomly crops a NumPy array (not a PIL image) to the specified image size.
    """
    h, w = arr.shape  # Ensure input is (H, W)
    crop_size = random.randint(
        int(image_size / max_crop_frac), int(image_size / min_crop_frac)
    )

    start_y = random.randint(0, h - crop_size)
    start_x = random.randint(0, w - crop_size)

    cropped = arr[start_y : start_y + crop_size, start_x : start_x + crop_size]
    return cropped