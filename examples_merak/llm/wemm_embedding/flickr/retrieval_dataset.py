"""Dataset interface for WeMM text-to-image retrieval evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    split: str
    rows: list[tuple[str, list[str]]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def limited(self, max_images: int) -> "RetrievalDataset":
        if max_images < 0:
            raise ValueError("max_images must be non-negative")
        if not max_images:
            return self
        return RetrievalDataset(
            name=self.name,
            split=self.split,
            rows=self.rows[:max_images],
            metadata=dict(self.metadata),
        )

    def flatten(self) -> tuple[list[str], list[str], list[int], list[list[int]]]:
        images = []
        captions = []
        caption_owners = []
        image_to_captions = []
        for image_index, (image, image_captions) in enumerate(self.rows):
            images.append(image)
            owned_caption_ids = []
            for caption in image_captions:
                caption_id = len(captions)
                captions.append(caption)
                caption_owners.append(image_index)
                owned_caption_ids.append(caption_id)
            image_to_captions.append(owned_caption_ids)
        return images, captions, caption_owners, image_to_captions

    def ordering_metadata(self) -> dict[str, Any]:
        images, captions, caption_owners, image_to_captions = self.flatten()
        return {
            "dataset": self.name,
            "split": self.split,
            "metadata": self.metadata,
            "images": images,
            "captions": captions,
            "caption_owners": caption_owners,
            "image_to_captions": image_to_captions,
        }


RetrievalDatasetLoader = Callable[[str | Path], RetrievalDataset]
_DATASET_LOADERS: dict[str, RetrievalDatasetLoader] = {}


def register_retrieval_dataset(
    name: str,
    loader: RetrievalDatasetLoader,
    *,
    force: bool = False,
) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Retrieval dataset name must be non-empty")
    if normalized in _DATASET_LOADERS and not force:
        raise ValueError(f"Retrieval dataset loader already registered: {normalized}")
    _DATASET_LOADERS[normalized] = loader


def available_retrieval_datasets() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_LOADERS))


def load_retrieval_dataset(name: str, dataset_dir: str | Path) -> RetrievalDataset:
    normalized = name.strip().lower()
    try:
        loader = _DATASET_LOADERS[normalized]
    except KeyError as exc:
        available = ", ".join(available_retrieval_datasets()) or "none"
        raise ValueError(f"Unknown retrieval dataset {name!r}; registered datasets: {available}") from exc
    dataset = loader(dataset_dir)
    if not dataset.rows:
        raise ValueError(f"Retrieval dataset {dataset.name!r} has no evaluation rows")
    return dataset


def _load_flickr30k(dataset_dir: str | Path) -> RetrievalDataset:
    try:
        from .flickr30k_utils import load_flickr30k
    except ImportError:
        from flickr30k_utils import load_flickr30k

    return RetrievalDataset(
        name="flickr30k",
        split="test",
        rows=load_flickr30k(dataset_dir),
        metadata={"dataset_dir": str(dataset_dir)},
    )


register_retrieval_dataset("flickr30k", _load_flickr30k)


__all__ = [
    "RetrievalDataset",
    "RetrievalDatasetLoader",
    "available_retrieval_datasets",
    "load_retrieval_dataset",
    "register_retrieval_dataset",
]
