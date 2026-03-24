import os
from typing import Iterable, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from abc import ABC, abstractmethod

# ============================================================
# 数据集注册表 & 工厂函数
# ============================================================

_DATASET_REGISTRY = {}


def register_dataset(name: str):
    """
    装饰器：注册一个数据集类，供 get_dataloader 使用。
    """
    def wrapper(cls):
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered.")
        _DATASET_REGISTRY[name] = cls
        return cls
    return wrapper


def get_dataloader(
    *,
    name: str,
    root: str,
    batch_size: int,
    image_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = True,
    device: Optional[str] = None,
    **dataset_kwargs,
):
    """
    工厂函数：根据注册名创建 DataLoader。

    现在的约定：
    - 可以直接用一个 dict / yml 做 ** 展开，只要 key 名字和这里一致即可。
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(_DATASET_REGISTRY.keys())}"
        )

    dataset_cls = _DATASET_REGISTRY[name]
    dataset = dataset_cls(
        root=root,
        image_size=image_size,
        **dataset_kwargs,     # 比如 extensions 会走到这里
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    if device is None:
        return dataloader
    else:
        return _DeviceDataLoader(dataloader, device)



class _DeviceDataLoader:
    """
    简单的 DataLoader 包装器：迭代时自动把 batch 搬到指定 device。
    """

    def __init__(self, dataloader: DataLoader, device: str):
        self.dataloader = dataloader
        self.device = torch.device(device)

    def __iter__(self):
        for batch in self.dataloader:
            # 这里只考虑 batch 是 Tensor 的情况，若以后你想返回 (img, label)，
            # 可以改成对每个元素递归 .to(self.device)
            yield batch.to(self.device)

    def __len__(self):
        return len(self.dataloader)


# ============================================================
# 抽象基类：约定子类必须实现 _load_image
# ============================================================

class BaseImageDataset(Dataset, ABC):
    """
    基类约定（contract）：
    ----------------------------------
    1. 子类构造函数签名形如：
         __init__(self, root: str, image_size: Optional[int] = None, **kwargs)
    2. 子类必须实现：
         def _load_image(self, path: str) -> torch.Tensor
       返回值必须是 [C, H, W] 的 Tensor，dtype=float32。
    3. 默认实现：
       - 递归收集文件路径（按照 extensions）
       - __len__ / __getitem__
    """

    def __init__(
        self,
        root: str,
        image_size: Optional[int] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        super().__init__()
        self.root = root
        self.extensions = tuple(e.lower() for e in extensions)
        self.image_size = image_size

        self.image_paths: List[str] = self._collect_image_paths(root)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under '{root}'.")

    # 子类可以复用或重写这个方法
    def _collect_image_paths(self, root: str) -> List[str]:
        image_paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.extensions:
                    full_path = os.path.join(dirpath, fname)
                    image_paths.append(full_path)
        image_paths.sort()
        return image_paths

    @abstractmethod
    def _load_image(self, path: str) -> torch.Tensor:
        """
        抽象方法：子类必须实现如何从 path 读取并返回 [C, H, W] 的 Tensor。
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        img = self._load_image(path)
        return img


# ============================================================
# 实现一个通用 jpg/png 读取数据集，并注册
# ============================================================

@register_dataset("image_folder")
class ImageFolderDataset(BaseImageDataset):
    """
    从目录中递归读取 jpg/png 图像，只返回图像 Tensor。
    """

    def __init__(
        self,
        root: str,
        image_size: Optional[int] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        # 调用基类构造，完成 root / extensions / 路径收集
        super().__init__(root=root, image_size=image_size, extensions=extensions)

        # 变换：可选 resize + ToTensor
        if image_size is not None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),  # [C, H, W], float32, [0,1]
            ])
        else:
            self.transform = T.ToTensor()

    def _load_image(self, path: str) -> torch.Tensor:
        """
        按照基类约定实现：给定 path，返回 [C, H, W] 的 Tensor。
        """
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img
