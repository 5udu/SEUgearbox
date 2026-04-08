from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import GramianAngularField
from scipy.io import savemat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GADF images and a serialized image dataset from segmented gearbox signals.",
    )
    parser.add_argument(
        "--segmented-data",
        type=Path,
        default=Path("artifacts/data/data_2.csv"),
        help="Path to the segmented CSV with the label column at the end.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("artifacts/gaf_images"),
        help="Directory to store generated JPG images.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/gaf_dataset.npz"),
        help="Serialized output dataset path (.npz or .mat).",
    )
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--paa-size", type=int, default=150)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--target-height", type=int, default=64)
    return parser.parse_args()


def generate_gadf_images(segmented_data: np.ndarray, image_dir: Path, image_size: int, paa_size: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    paa = PiecewiseAggregateApproximation(window_size=None, output_size=paa_size)
    gadf = GramianAngularField(
        image_size=image_size,
        method="difference",
        sample_range=(-1, 1),
        overlapping=False,
    )

    for index, sample in enumerate(segmented_data, start=1):
        sample_paa = paa.transform(sample.reshape(1, -1))
        image = gadf.fit_transform(sample_paa)[0]

        figure = plt.figure(figsize=(6.25, 6.25), dpi=100)
        plt.imshow(image, cmap="jet", origin="lower", aspect="auto")
        plt.axis("off")
        plt.gca().set_position([0, 0, 1, 1])
        plt.savefig(image_dir / f"{index:04d}.jpg", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(figure)


def serialize_images(
    image_dir: Path,
    labels: np.ndarray,
    output_path: Path,
    target_size: tuple[int, int],
) -> None:
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        warnings.warn(f"No jpg files found in {image_dir}", UserWarning)
        return
    if len(image_files) != len(labels):
        raise ValueError(
            f"Image count ({len(image_files)}) does not match label count ({len(labels)})"
        )

    images = []
    for image_path in image_files:
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(np.asarray(image.resize(target_size), dtype=np.float32) / 255.0)

    images_array = np.asarray(images)
    labels_array = np.asarray(labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mat":
        savemat(output_path, {"resizeimg": images_array, "labels": labels_array}, do_compression=True)
    else:
        np.savez(output_path, images=images_array, labels=labels_array)


def main() -> None:
    args = parse_args()
    segmented = pd.read_csv(args.segmented_data)
    labels = segmented.iloc[:, -1].to_numpy()
    data = segmented.iloc[:, :-1].to_numpy(dtype=np.float64)

    generate_gadf_images(
        segmented_data=data,
        image_dir=args.image_dir,
        image_size=args.image_size,
        paa_size=args.paa_size,
    )
    serialize_images(
        image_dir=args.image_dir,
        labels=labels,
        output_path=args.output_path,
        target_size=(args.target_width, args.target_height),
    )
    print(f"Saved GADF images to {args.image_dir}")
    print(f"Saved serialized dataset to {args.output_path}")


if __name__ == "__main__":
    main()
