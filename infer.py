"""Launch the supplied test.py with user images and complete output folders."""

import argparse
from pathlib import Path
import subprocess
import sys

from PIL import Image


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate one composite using the original CGANet inference code."
    )
    parser.add_argument("--foreground", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=root / "experiments")
    parser.add_argument("--vgg", type=Path, default=root / "models/vgg_normalised.pth")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    images = {
        "foreground": args.foreground.expanduser().resolve(),
        "mask": args.mask.expanduser().resolve(),
        "background": args.background.expanduser().resolve(),
    }
    weights_dir = args.weights_dir.expanduser().resolve()
    weights = {
        "vgg": args.vgg.expanduser().resolve(),
        "decoder": weights_dir / "decoder_iter_80000.pth",
        "PSF": weights_dir / "PSF_iter_80000.pth",
        "NFFM": weights_dir / "NFFM_iter_80000.pth",
        "FEFM": weights_dir / "FEFM_iter_80000.pth",
    }
    for name, path in list(images.items()) + list(weights.items()):
        if not path.is_file():
            parser.error("{} file does not exist: {}".format(name, path))

    try:
        with Image.open(images["foreground"]) as image:
            foreground_size = image.size
            if image.mode != "RGB":
                parser.error("The foreground image must be RGB.")
        with Image.open(images["mask"]) as image:
            if image.mode not in ("1", "L"):
                parser.error("The mask must be a single-channel image (mode 1 or L).")
            if image.size != foreground_size:
                parser.error("The foreground and mask must have the same dimensions.")
        with Image.open(images["background"]) as image:
            width, height = image.size
            if image.mode != "RGB":
                parser.error("The background image must be RGB.")
    except (OSError, ValueError) as error:
        parser.error("Cannot read an input image: {}".format(error))

    output = args.output.expanduser().resolve()
    for subdirectory in ("images", "original", "mask"):
        (output / subdirectory).mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(root / "test.py"),
        "--use_examples", "0",
        "--fore", str(images["foreground"]),
        "--mask", str(images["mask"]),
        "--back", str(images["background"]),
        "--zoomSize", "1",
        "--Vertical", "0",
        "--Horizontal", "0",
        "--Left", "0",
        "--Top", "0",
        "--Right", str(width - 1),
        "--Bottom", str(height - 1),
        "--vgg", str(weights["vgg"]),
        "--decoder", str(weights["decoder"]),
        "--PSF", str(weights["PSF"]),
        "--NFFM", str(weights["NFFM"]),
        "--FEFM", str(weights["FEFM"]),
        "--save_ext", ".png",
        "--output", str(output),
    ]
    result = subprocess.run(command, cwd=str(root))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
