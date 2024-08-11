from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable
import dataclasses
from dataclasses import dataclass
from enum import Enum
from glob import glob
import logging
from math import ceil
import os.path
from pathlib import Path
import sys
from typing import Any

from colour import Color
import numpy as np
from PIL import Image, ImageOps


logger = logging.getLogger(__name__)
logging.basicConfig(style='{', format='{levelname}: {message}')


class ExistingBorderHandling(Enum):
    SKIP = 'skip'       # Skip file
    ADD = 'add'         # Add a border anyway
    REPLACE = 'replace' # Remove old border and add new one

    # For argparse help output.
    def __str__(self):
        return self.value


@dataclass(frozen=True)
class BorderConfig:
    colour: Color
    baseline_size: float    # Proportional to image size
    min_aspect_ratio: float
    max_aspect_ratio: float


@dataclass(frozen=True)
class AppConfig:
    input_path: str     # File name or glob
    existing_border_handling: ExistingBorderHandling
    border: BorderConfig
    output_directory: Path | None  # If none, output to input directory
    output_file_name_suffix: str
    allow_overwrite: bool
    dry_run: bool
    verbose: bool


class AppError(Exception):
    """A fatal, application-specific error."""


def get_config(args: list[str]) -> AppConfig:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', type=str, help='File path, directory path, or path glob to process.')
    # TODO? allow excluding files
    parser.add_argument('--existing-border', type=ExistingBorderHandling,
        choices=list(ExistingBorderHandling),
        default=ExistingBorderHandling.SKIP,
        help='How to handle images with existing borders. '
            f'{ExistingBorderHandling.SKIP}: Don\'t process the image. '
            f'{ExistingBorderHandling.ADD}: Add the new border anyway. '
            f'{ExistingBorderHandling.REPLACE}: Replace the existing border.')
    parser.add_argument('--border-colour', type=Color, default='white',
        help='Border colour, as a W3C colour name.')
    parser.add_argument('--border-size', type=float, default=0.1,
        help='Baseline border size, as a proportion of the average image dimension.')
    parser.add_argument('--min-aspect', type=float, default=0.84, help='Minimum aspect ratio.')
    parser.add_argument('--max-aspect', type=float, default=1.2, help='Maximum aspect ratio.')
    parser.add_argument('--output-dir', type=Path, default=None,
        help='Output directory path. Defaults to output in the same directory as the input.')
    parser.add_argument('--output-suffix', type=str, default='-border', help='Output file name suffix.')
    parser.add_argument('--overwrite', action='store_true', default=False,
        help='Allow overwriting files which already exist.')
    parser.add_argument('--dry-run', action='store_true', default=False,
        help='Simulate the operation without writing any files.')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='Print more information about the operation')

    parsed = parser.parse_args(args)

    if parsed.min_aspect > parsed.max_aspect:
        raise AppError('Minimum aspect ratio must be <= maximum aspect ratio')

    return AppConfig(
        input_path=parsed.files,
        existing_border_handling=parsed.existing_border,
        border=BorderConfig(
            colour=parsed.border_colour,
            baseline_size=parsed.border_size,
            min_aspect_ratio=parsed.min_aspect,
            max_aspect_ratio=parsed.max_aspect
        ),
        output_directory=parsed.output_dir,
        output_file_name_suffix=parsed.output_suffix,
        allow_overwrite=parsed.overwrite,
        dry_run=parsed.dry_run,
        verbose=parsed.verbose)


def log_config(config: AppConfig) -> None:
    for field in dataclasses.fields(config):
        logger.debug(f'Config: {field.name}: {getattr(config, field.name)}')


def get_input_file_paths(name_or_glob: str) -> list[Path]:
    if os.path.isfile(name_or_glob):
        # If path is a file, process just that file.
        logger.debug(f'Input path is a file')
        return [Path(name_or_glob)]
    elif os.path.isdir(name_or_glob):
        # If path is a directory, process all files in that directory.
        logger.debug(f'Input path is a directory, will process all contained files')
        return [p for p in Path(name_or_glob).iterdir() if p.is_file()]
    else:
        # Otherwise, find files via glob.
        logger.debug(f'Input path is a glob, will process all matching files')
        return [Path(p) for p in glob(name_or_glob) if os.path.isfile(p)]


def validate_output_paths(output_paths: Iterable[Path], allow_overwrite: bool) -> None:
    if not allow_overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            existing_str = ','.join(f'\'{path}\'' for path in existing)
            raise AppError(AppError(f'Would overwrite existing files: {existing_str}'))


@dataclass(frozen=True)
class BorderSize:
    # Size in pixels of the border on each side.
    top: int
    bottom: int
    left: int
    right: int

    @property
    def pil_tuple(self) -> tuple[int, int, int, int]:
        """As used in various PIL functions: left, top, right, bottom."""

        return (self.left, self.top, self.right, self.bottom)

    @property
    def all_sides(self) -> bool:
        """Returns true if there is a border on all sides."""

        return all(b > 0 for b in (self.top, self.bottom, self.left, self.right))


# Relative diff above which pixels are considered to be different colours for the purpose of border detection.
# TODO? make this configurable
BORDER_DIFF_COLOUR_THRESHOLD = 0.05

# Proportion of pixels in a border size that are allowed to be different.
# TODO? make this configurable
BORDER_DIFF_PROPORTION_THRESHOLD = 0.01


def detect_border(image: Image.Image, channel_diff_threshold: float = BORDER_DIFF_COLOUR_THRESHOLD,
        pixel_count_threshold: float = BORDER_DIFF_PROPORTION_THRESHOLD) -> BorderSize:
    """Infers the size of an image's border from pixel values.
        The border must be uniform colour on all sides."""

    data = np.array(image)
    # Shape is (height, width, channels)

    # Reference top left pixel, to which colours are compared
    ref_colour = data[0, 0].astype(float)
    # Relative differences are a proportion of the max pixel value.
    diff_ref = np.max(data, axis=(0, 1))

    def is_same_colour(pixels: np.ndarray) -> bool:
        # Cast to avoid clipping on unsigned pixel types.
        pixels = pixels.astype(float)
        diffs = pixels - ref_colour
        # Each channel must be within the threshold.
        different = np.abs(diffs) > channel_diff_threshold * diff_ref
        total_pixels = pixels.shape[0] * pixels.shape[1]
        different_proportion = np.count_nonzero(different) / total_pixels
        overall_same = different_proportion <= pixel_count_threshold
        return overall_same

    def find_border(axis: int, reverse: bool) -> int:
        depth = 1
        while True:
            index = -depth if reverse else depth - 1
            side = data[index, :] if axis == 0 else data[:, index]
            if not is_same_colour(side):
                return depth - 1
            depth += 1

    return BorderSize(
        top=find_border(0, False),
        bottom=find_border(0, True),
        left=find_border(1, False),
        right=find_border(1, True)
    )


def calculate_new_border_size(size: tuple[int, int], min_aspect: float, max_aspect: float, baseline_border_size: float) \
        -> BorderSize:
    """Calculates the desired border size in pixels, based on aspect ratio bounds."""

    # Try to create a constant-sized border according to baseline_border_size.
    # If the resulting image falls outside the aspect ratio bounds, expand the border in one dimension such that the
    # aspect ratio becomes within the bounds.

    assert min_aspect <= max_aspect

    width, height = size
    avg_dim = np.mean(size)

    border_width_pixels = int(round(baseline_border_size * avg_dim))
    border_height_pixels = int(round(baseline_border_size * avg_dim))

    def new_width() -> int: return width + border_width_pixels * 2
    def new_height() -> int: return height + border_height_pixels * 2
    def new_aspect() -> float: return new_width() / new_height()
    
    if new_aspect() < min_aspect:
        # Too tall, add to left/right borders.
        desired_width = min_aspect * new_height()
        border_width_pixels = ceil((desired_width - size[0]) / 2)
    elif new_aspect() > max_aspect:
        # Too wide, add to top/bottom borders.
        desired_height = new_width() / max_aspect
        border_height_pixels = ceil((desired_height - size[1]) / 2)
    
    return BorderSize(
        top=border_height_pixels, bottom=border_height_pixels,
        left=border_width_pixels, right=border_width_pixels)


def apply_new_border(image: Image.Image, config: BorderConfig) \
        -> Image.Image:
    border_size = calculate_new_border_size(
        image.size, config.min_aspect_ratio, config.max_aspect_ratio, config.baseline_size)
    new_image = ImageOps.expand(image, border=border_size.pil_tuple, fill=config.colour.get_hex_l())
    logger.debug(
        f'New dimensions {new_image.width}x{new_image.height} (aspect ratio {new_image.width / new_image.height:.2f})')
    return new_image


def remove_border(image: Image.Image) -> Image.Image:
    existing_border = detect_border(image)
    new_size = existing_border.pil_tuple
    new_image = image.crop(new_size)
    return new_image


def get_image_write_params(image: Image.Image) -> dict[str, Any]:
    write_params: dict[str, Any] = {
        # Preserve metadata
        'icc_profile': image.info.get('icc_profile'),   # Colour profile
        'exif': image.getexif()     # Camera info and such
    }
    if image.format == 'JPEG':
        # Default JPG writing settings are garbage. Aim to preserve quality as much as possible.
        write_params |= {
            'quality': 95,
            'subsampling': 0,
            'optimize': True
        }
    return write_params


def process_image(input_path: Path, output_path: Path, config: AppConfig) -> None:
    logger.info(f'Processing \'{input_path}\'')
    try:
        image = Image.open(input_path)
    except Image.UnidentifiedImageError:
        # Ignore files which are not images.
        logger.info(f'Ignored non-image file')
        return

    # We're trying to preserve as much as possible from the original image, so save the info now in case it's changed.
    write_params = get_image_write_params(image)

    match config.existing_border_handling:
        case ExistingBorderHandling.ADD:
            image = apply_new_border(image, config.border)    
        case ExistingBorderHandling.SKIP:
            if detect_border(image).all_sides:
                logger.info('Skipping, border already present')
                return
            else:
                image = apply_new_border(image, config.border)
        case ExistingBorderHandling.REPLACE:
            image = remove_border(image)
            image = apply_new_border(image, config.border)
        case _: # type: ignore
            raise AssertionError('Unhandled ExistingBorderHandling mode')
    
    if config.dry_run:
        logger.info(f'Dry run: Would save image to \'{output_path}\'')
    else:
        # Create the output directory if required.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not config.allow_overwrite and output_path.exists():
            raise AppError(f'Would overwrite existing file: \'{output_path}\'')
        image.save(output_path, **write_params)
        logger.info(f'Saved image to \'{output_path}\'')


def get_output_image_path(input_path: Path, output_directory: Path | None, output_suffix: str) -> Path:
    out_dir = output_directory or input_path.parent
    name = input_path.name
    if output_suffix:
        parts = name.split('.')
        parts[0] = parts[0] + output_suffix
        name = '.'.join(parts)
    return out_dir / name


def main(args: list[str]):
    try:
        config = get_config(args)

        if config.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        log_config(config)

        input_file_paths = get_input_file_paths(config.input_path)
        if not input_file_paths:
            logger.info('No files to process')
            return

        output_file_paths = [
            get_output_image_path(p, config.output_directory, config.output_file_name_suffix) for p in input_file_paths]

        validate_output_paths(output_file_paths, config.allow_overwrite)

        for input_path, output_path in zip(input_file_paths, output_file_paths):
            process_image(input_path, output_path, config)
        
        logger.info('Success')
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
