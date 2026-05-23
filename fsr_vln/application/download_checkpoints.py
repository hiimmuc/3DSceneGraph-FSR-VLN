"""Download model checkpoints if they do not already exist.
- Open CLIP ViT-L/14: https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K
- SAM ViT-H:          https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

import urllib.request
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Registry: filename → (url, human-readable size)
# ---------------------------------------------------------------------------
CHECKPOINT_URLS: dict[str, tuple[str, str]] = {
    "mobileclip2_s4.pt": (
        "https://huggingface.co/apple/MobileCLIP2-S4/resolve/main/mobileclip2_s4.pt",
        "~350 MB",
    ),
    "open_clip_pytorch_model.bin": (
        "https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin?download=true",
        "~800 MB",
    ),
    "sam_vit_h_4b8939.pth": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "~2.5 GB",
    ),
    "sam_vit_b_01ec64.pth": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "~375 MB",
    ),
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1_048_576
        print(f"\r  {mb:.1f} MB  ({pct:.1f}%)", end="", flush=True)


def download_checkpoint(dest: Path) -> None:
    """Download a single checkpoint to *dest* if it does not already exist."""
    filename = dest.name
    if filename not in CHECKPOINT_URLS:
        raise KeyError(
            f"No download URL registered for '{filename}'.\n"
            f"Please download it manually and place it at: {dest}"
        )

    if dest.exists():
        print(f"[checkpoint] Already exists, skipping: {dest}")
        return

    url, size = CHECKPOINT_URLS[filename]
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] Downloading {filename} ({size})\n  → {dest}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # newline after progress
        print(f"[checkpoint] Saved {filename}")
    except Exception as exc:
        dest.unlink(missing_ok=True)  # remove partial file
        raise RuntimeError(
            f"Failed to download {filename}: {exc}\n"
            f"Download manually:\n  wget '{url}' -O {dest}"
        ) from exc


def ensure_checkpoints(checkpoint_paths: list[Union[str, Path]]) -> None:
    """Check each path and download the file if it is missing."""
    for p in checkpoint_paths:
        download_checkpoint(Path(p))


# ---------------------------------------------------------------------------
# CLI entry-point — download all registered checkpoints to ./checkpoints/
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download HoloAgent model checkpoints.")
    parser.add_argument(
        "--dir",
        default="checkpoints",
        help="Destination directory (default: ./checkpoints)",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dir)
    for filename in CHECKPOINT_URLS:
        download_checkpoint(dest_dir / filename)
