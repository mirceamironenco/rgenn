import os
import pathlib
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from scipy import io
from torch.utils.data.dataset import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from rgenn.utils import get_local_rank


class AffnistDataset(VisionDataset):
    urls = None
    raw_files = None
    processed_files = None  # (train, test)

    def __init__(
        self,
        root: pathlib.Path | str,
        train: bool,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform

        local_rank = get_local_rank()
        if download and local_rank == 0:
            self.download()

        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

        if not self._check_exists():
            raise RuntimeError("affNIST dataset not downloaded and download=False.")

        file = self.processed_files[0] if train else self.processed_files[1]
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, file))

    def download(self):
        if self._check_exists():
            return

        assert self.urls is not None
        assert self.raw_files is not None
        assert self.processed_files is not None

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for url in self.urls:
            zip_file = url.rpartition("/")[-1]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=zip_file
            )

        raw_files = [
            os.path.join(self.raw_folder, raw_file) for raw_file in self.raw_files
        ]

        for raw_file, processed_file in zip(raw_files, self.processed_files):
            data = io.loadmat(raw_file, simplify_cells=True)["affNISTdata"]
            images = data["image"].transpose().reshape(-1, 40, 40)
            labels = data["label_int"].astype(np.int64)
            torch.save(
                (images, labels), os.path.join(self.processed_folder, processed_file)
            )

    def _check_exists(self):
        return all(
            os.path.exists(os.path.join(self.processed_folder, file))
            for file in self.processed_files
        )

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def n_classes(self):
        return 10

    @property
    def n_channels(self):
        return 1

    @property
    def img_size(self):
        return 40, 40

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)


class AffnistUntransformed(AffnistDataset):
    urls = (
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/training.mat.zip",
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/test.mat.zip",
    )
    raw_files = ("training.mat", "test.mat")
    processed_files = ("train.pt", "test.pt")


class AffnistTransformed(AffnistDataset):
    """
    NB: Returns the validation set for train=True, as the training set is too
    large and not used for training since most models are trained on
    untransformed affNIST to demonstrate generalization.
    """

    urls = (
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/validation.mat.zip",
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip",
    )
    raw_files = ("validation.mat", "test.mat")
    processed_files = ("val.pt", "test.pt")


class HomnistTest(Dataset):
    # For some reason the keys in the .mat were renamed so ...
    # TODO: Subclass and pass 'image_data' and 'labels' keys as parameters
    def __init__(self, root="datasets", transform=None, **kwargs):
        super().__init__()
        self.root = root
        self.transform = transform if transform is not None else lambda x: x
        self.data_path = pathlib.Path.cwd() / root / "homNIST"
        self.split = "homNIST_test.mat"
        self.data, self.targets = self._load_data()

    def _load_data(self):
        # https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
        path = self.data_path / self.split
        data = io.loadmat(path, simplify_cells=True)
        images = torch.from_numpy(data["img_data"]).to(torch.float32).unsqueeze(1)
        labels = data["labels"]
        return images, labels

    def _get_matrices(self):
        path = self.data_path / self.split
        data = io.loadmat(path, simplify_cells=True)
        matrices = data["hom_matrices"].transpose().reshape(-1, 3, 3)
        return matrices

    @property
    def n_classes(self):
        return 10

    @property
    def n_channels(self):
        return 1

    @property
    def img_size(self):
        return 40, 40

    def __getitem__(self, index):
        img, target = self.transform(self.data[index]), int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.targets)
