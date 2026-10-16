# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:55:18 2026

@author: Gail McConnell, SIPBS, University of Strathclyde, UK
"""

from __future__ import annotations

"""
Background homogenisation for microscopy images while preserving protected
foreground pixels unchanged in memory.

Supported formats:
    PNG, JPG, JPEG, TIF, TIFF

The output is saved using the same NumPy datatype as the input:
    uint8  -> uint8
    uint16 -> uint16
    int16  -> int16
    float32 -> float32
    float64 -> float64

Dependencies:
    numpy
    scipy
    imageio
    pillow
    tifffile

Install with:
    conda install -c conda-forge numpy scipy imageio pillow tifffile

Usage:
    1. Set INPUT and OUTPUT in the USER SETTINGS section.
    2. Run the script in Spyder, VS Code, PyCharm, or from a terminal.
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    label,
    sobel,
)


# ============================================================================
# USER SETTINGS
# ============================================================================

INPUT = "example_input.tif"
OUTPUT = "example_output.tif"


# Expand the protected regions of the specimen mask by this many pixels.
# Increase this if true specimen edges or halos are being altered.
# Decrease this if background artefacts remain immediately around cells.
DILATION_PIXELS = 4 #default value=4


# Foreground detection thresholds.
#
# Lower values protect more pixels.
# Higher values protect fewer pixels.
LOCAL_CONTRAST_THRESHOLD = 3.5 #default value=3.5
GRADIENT_THRESHOLD = 4 #default value=4
DARKNESS_THRESHOLD = 3 #default value=3


# Remove automatically detected foreground components smaller than this area.
# Increase this if isolated TIFF background speckles are being protected.
MINIMUM_OBJECT_SIZE = 30 #default value=30


# Spatial scales used to separate biological structures from background.
SMALL_SIGMA = 1.5
MEDIUM_SIGMA = 8.0
LARGE_SIGMA = 80.0


# Background output mode:
#
#     "constant" = all unprotected pixels are replaced by one background value
#     "smooth"   = unprotected pixels are replaced by a smoothly varying field
#
# Use "constant" when the goal is complete removal of a tiled background.
BACKGROUND_MODE = "constant"


# Used only when BACKGROUND_MODE = "smooth".
SMOOTH_BACKGROUND_SIGMA = 35.0


# Diagnostic output.
SAVE_MASK = True
SAVE_BACKGROUND_ESTIMATE = True


# TIFF compression.
#
# None preserves maximum compatibility.
# You may also try "zlib" for lossless compression.
TIFF_COMPRESSION = None


# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ============================================================================
# IMAGE INPUT
# ============================================================================

def load_image(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load a grayscale, RGB, or RGBA image.

    Parameters
    ----------
    filename:
        Input PNG, JPEG, or TIFF filename.

    Returns
    -------
    original:
        Input image in its original datatype.

    working:
        Floating-point grayscale or RGB image used during processing.

    alpha:
        Original alpha channel, when present; otherwise None.
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{path}"
        )

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported input format: {path.suffix}\n"
            "Supported formats are PNG, JPG, JPEG, TIF, and TIFF."
        )

    if path.suffix.lower() in {".tif", ".tiff"}:
        original = tifffile.imread(path)
    else:
        original = iio.imread(path)

    if original.ndim == 2:
        working = original.astype(np.float32)
        alpha = None

    elif original.ndim == 3 and original.shape[2] == 3:
        working = original.astype(np.float32)
        alpha = None

    elif original.ndim == 3 and original.shape[2] == 4:
        working = original[:, :, :3].astype(np.float32)
        alpha = original[:, :, 3].copy()

    else:
        raise ValueError(
            f"Unsupported image shape: {original.shape}\n"
            "Expected a 2D grayscale image, RGB image, or RGBA image."
        )

    return original, working, alpha


def make_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Create a grayscale analysis image without altering the source image.
    """
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)

    return (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    ).astype(np.float32)


# ============================================================================
# DATATYPE AND BIT-DEPTH PRESERVATION
# ============================================================================

def convert_to_original_dtype(
    image: np.ndarray,
    original_dtype: np.dtype,
) -> np.ndarray:
    """
    Convert processed floating-point data back to the input datatype.

    Integer data are clipped to their valid range and rounded.
    Floating-point data are cast directly without intensity rescaling.
    """
    dtype = np.dtype(original_dtype)

    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)

        return np.clip(
            np.rint(image),
            limits.min,
            limits.max,
        ).astype(dtype)

    if np.issubdtype(dtype, np.floating):
        return image.astype(dtype)

    if np.issubdtype(dtype, np.bool_):
        return image.astype(bool)

    raise TypeError(
        f"Unsupported input datatype: {dtype}"
    )


def describe_dtype(dtype: np.dtype) -> str:
    """
    Return a readable datatype and bit-depth description.
    """
    dtype = np.dtype(dtype)

    return f"{dtype.name} ({dtype.itemsize * 8}-bit)"


# ============================================================================
# IMAGE OUTPUT
# ============================================================================

def save_image(
    filename: str | Path,
    processed: np.ndarray,
    original_dtype: np.dtype,
    alpha: np.ndarray | None = None,
) -> None:
    """
    Save the processed image using the input datatype.

    TIFF images are written with tifffile.
    PNG and JPEG images are written with imageio.

    JPEG is inherently lossy and cannot preserve pixel values exactly.
    """
    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format: {suffix}\n"
            "Use PNG, JPG, JPEG, TIF, or TIFF."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    output = convert_to_original_dtype(
        processed,
        original_dtype,
    )

    if alpha is not None:
        alpha_output = convert_to_original_dtype(
            alpha,
            original_dtype,
        )

        if output.ndim != 3 or output.shape[2] != 3:
            raise ValueError(
                "An alpha channel can only be added to an RGB output image."
            )

        output = np.dstack(
            (output, alpha_output)
        )

    if suffix in {".tif", ".tiff"}:
        if output.ndim == 2:
            photometric = "minisblack"
        elif output.ndim == 3 and output.shape[2] in (3, 4):
            photometric = "rgb"
        else:
            photometric = None

        tifffile.imwrite(
            path,
            output,
            photometric=photometric,
            compression=TIFF_COMPRESSION,
        )

    elif suffix == ".png":
        if output.dtype not in (
            np.uint8,
            np.uint16,
        ):
            raise ValueError(
                "PNG output is normally limited to uint8 or uint16 data.\n"
                f"The input datatype is {output.dtype}.\n"
                "Use TIFF output to preserve floating-point or signed data."
            )

        iio.imwrite(
            path,
            output,
        )

    else:
        print(
            "Warning: JPEG is lossy and generally limited to 8-bit data. "
            "Foreground values will not remain numerically identical after "
            "JPEG compression."
        )

        jpeg_output = prepare_jpeg_output(output)

        iio.imwrite(
            path,
            jpeg_output,
            quality=95,
        )


def prepare_jpeg_output(image: np.ndarray) -> np.ndarray:
    """
    Convert data to uint8 for JPEG output.

    This function is used only because JPEG does not support scientific
    integer and floating-point datatypes.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)

    finite = np.isfinite(image_float)

    if not np.any(finite):
        return np.zeros(
            image_float.shape,
            dtype=np.uint8,
        )

    minimum = float(np.min(image_float[finite]))
    maximum = float(np.max(image_float[finite]))

    if maximum <= minimum:
        return np.zeros(
            image_float.shape,
            dtype=np.uint8,
        )

    scaled = (
        255.0
        * (image_float - minimum)
        / (maximum - minimum)
    )

    return np.clip(
        np.rint(scaled),
        0,
        255,
    ).astype(np.uint8)


def scale_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Scale arbitrary data to uint8 for diagnostic preview images.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    minimum = float(np.min(image[finite]))
    maximum = float(np.max(image[finite]))

    if maximum <= minimum:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    scaled = (
        255.0
        * (image - minimum)
        / (maximum - minimum)
    )

    return np.clip(
        np.rint(scaled),
        0,
        255,
    ).astype(np.uint8)


# ============================================================================
# ROBUST THRESHOLDING
# ============================================================================

def robust_z_score(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to robust standard-deviation units using the MAD.

    MAD is the median absolute deviation.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    median = float(
        np.median(image)
    )

    mad = float(
        np.median(
            np.abs(image - median)
        )
    )

    robust_sigma = 1.4826 * mad

    if robust_sigma < np.finfo(np.float32).eps:
        robust_sigma = float(
            np.std(image)
        )

    if robust_sigma < np.finfo(np.float32).eps:
        robust_sigma = 1.0

    return (
        image - median
    ) / robust_sigma


def remove_small_components(
    mask: np.ndarray,
    minimum_size: int,
) -> np.ndarray:
    """
    Remove connected foreground components below a specified area.
    """
    if minimum_size <= 0:
        return mask.astype(bool)

    labelled, number = label(mask)

    if number == 0:
        return mask.astype(bool)

    component_sizes = np.bincount(
        labelled.ravel()
    )

    retain = (
        component_sizes >= int(minimum_size)
    )

    retain[0] = False

    return retain[labelled]


# ============================================================================
# FOREGROUND MASK
# ============================================================================

def create_protected_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect specimen structures using local contrast, edges, and darkness.

    Broad background variation is removed from the detection process so that
    tiled intensity variation is less likely to be classified as foreground.
    """
    image = image.astype(
        np.float32,
        copy=False,
    )

    fine = gaussian_filter(
        image,
        sigma=SMALL_SIGMA,
    )

    medium = gaussian_filter(
        image,
        sigma=MEDIUM_SIGMA,
    )

    broad = gaussian_filter(
        image,
        sigma=LARGE_SIGMA,
    )

    local_detail = np.abs(
        fine - medium
    )

    relative_darkness = (
        broad - fine
    )

    gradient_x = sobel(
        fine,
        axis=1,
    )

    gradient_y = sobel(
        fine,
        axis=0,
    )

    gradient = np.hypot(
        gradient_x,
        gradient_y,
    )

    detail_z = robust_z_score(
        local_detail
    )

    darkness_z = robust_z_score(
        relative_darkness
    )

    gradient_z = robust_z_score(
        gradient
    )

    detail_mask = (
        detail_z > LOCAL_CONTRAST_THRESHOLD
    )

    darkness_mask = (
        darkness_z > DARKNESS_THRESHOLD
    )

    gradient_mask = (
        gradient_z > GRADIENT_THRESHOLD
    )

    protected = (
        detail_mask
        | darkness_mask
        | gradient_mask
    )

    protected = binary_closing(
        protected,
        iterations=2,
    )

    protected = binary_fill_holes(
        protected
    )

    protected = remove_small_components(
        protected,
        MINIMUM_OBJECT_SIZE,
    )

    if DILATION_PIXELS > 0:
        protected = binary_dilation(
            protected,
            iterations=int(DILATION_PIXELS),
        )

    return protected.astype(bool)


# ============================================================================
# BACKGROUND RECONSTRUCTION
# ============================================================================

def normalized_gaussian_background(
    image: np.ndarray,
    protected_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Estimate a smooth background while excluding protected pixels.

    This uses normalized convolution:

        Gaussian(image * valid) / Gaussian(valid)
    """
    valid = (
        ~protected_mask
    ).astype(np.float32)

    numerator = gaussian_filter(
        image * valid,
        sigma=sigma,
    )

    denominator = gaussian_filter(
        valid,
        sigma=sigma,
    )

    minimum_weight = 1e-6

    background = numerator / np.maximum(
        denominator,
        minimum_weight,
    )

    return background.astype(
        np.float32
    )


def calculate_background_target(
    image: np.ndarray,
    protected_mask: np.ndarray,
) -> float:
    """
    Calculate the median value of the unprotected background.
    """
    background_pixels = (
        ~protected_mask
    )

    if not np.any(background_pixels):
        raise RuntimeError(
            "No unprotected background pixels remain."
        )

    return float(
        np.median(
            image[background_pixels]
        )
    )


def homogenise_channel(
    channel: np.ndarray,
    protected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Homogenise one grayscale or colour channel.
    """
    background_target = calculate_background_target(
        channel,
        protected_mask,
    )

    corrected = channel.copy()

    if BACKGROUND_MODE.lower() == "constant":
        estimated_background = np.full(
            channel.shape,
            background_target,
            dtype=np.float32,
        )

        corrected[~protected_mask] = (
            background_target
        )

    elif BACKGROUND_MODE.lower() == "smooth":
        estimated_background = normalized_gaussian_background(
            channel,
            protected_mask,
            sigma=SMOOTH_BACKGROUND_SIGMA,
        )

        field_median = float(
            np.median(
                estimated_background[
                    ~protected_mask
                ]
            )
        )

        if abs(field_median) > np.finfo(np.float32).eps:
            estimated_background *= (
                background_target
                / field_median
            )

        corrected[~protected_mask] = (
            estimated_background[
                ~protected_mask
            ]
        )

    else:
        raise ValueError(
            'BACKGROUND_MODE must be "constant" or "smooth".'
        )

    # Exact copy from the original working array.
    corrected[protected_mask] = (
        channel[protected_mask]
    )

    return corrected, estimated_background


def homogenise_background(
    image: np.ndarray,
    protected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Homogenise only unprotected background pixels.

    Protected specimen pixels are copied from the source working array.
    """
    if image.ndim == 2:
        corrected, estimated = homogenise_channel(
            image,
            protected_mask,
        )

    elif image.ndim == 3 and image.shape[2] == 3:
        corrected = image.copy()
        estimated_channels = []

        for channel_index in range(3):
            corrected_channel, estimated_channel = homogenise_channel(
                image[:, :, channel_index],
                protected_mask,
            )

            corrected[:, :, channel_index] = corrected_channel

            estimated_channels.append(
                estimated_channel
            )

        estimated = np.stack(
            estimated_channels,
            axis=2,
        )

        corrected[protected_mask, :] = (
            image[protected_mask, :]
        )

    else:
        raise ValueError(
            "Expected a grayscale image or a three-channel RGB image."
        )

    return corrected, estimated


# ============================================================================
# DIAGNOSTIC OUTPUT
# ============================================================================

def diagnostic_path(
    output_path: str | Path,
    ending: str,
) -> Path:
    """
    Create a diagnostic PNG filename beside the main output.
    """
    output_path = Path(output_path)

    return output_path.with_name(
        output_path.stem
        + ending
        + ".png"
    )


def save_diagnostics(
    output_path: str | Path,
    protected_mask: np.ndarray,
    estimated_background: np.ndarray,
) -> None:
    """
    Save a foreground mask and background preview.
    """
    if SAVE_MASK:
        mask_path = diagnostic_path(
            output_path,
            "_protected_mask",
        )

        iio.imwrite(
            mask_path,
            protected_mask.astype(np.uint8) * 255,
        )

        print(
            f"Protected mask saved to:\n{mask_path}"
        )

    if SAVE_BACKGROUND_ESTIMATE:
        background_path = diagnostic_path(
            output_path,
            "_estimated_background",
        )

        iio.imwrite(
            background_path,
            scale_to_uint8(
                estimated_background
            ),
        )

        print(
            "Estimated background preview saved to:\n"
            f"{background_path}"
        )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_preserved_pixels(
    original_working: np.ndarray,
    corrected_working: np.ndarray,
    protected_mask: np.ndarray,
) -> None:
    """
    Confirm that protected pixels are identical before datatype conversion.
    """
    if original_working.ndim == 2:
        equal = np.array_equal(
            original_working[protected_mask],
            corrected_working[protected_mask],
        )
    else:
        equal = np.array_equal(
            original_working[protected_mask, :],
            corrected_working[protected_mask, :],
        )

    if not equal:
        raise RuntimeError(
            "Protected pixels were unexpectedly changed during processing."
        )

    print(
        "Protected foreground pixels verified unchanged before saving."
    )


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main() -> None:
    """
    Run the complete PROTECT workflow.
    """
    print(
        "PROTECT background homogenisation"
    )
    print(
        "--------------------------------"
    )

    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    print(
        f"Input:\n{input_path}"
    )
    print(
        f"Output:\n{output_path}"
    )

    original, working, alpha = load_image(
        input_path
    )

    print(
        f"Input image shape: {original.shape}"
    )
    print(
        "Input datatype and bit-depth: "
        f"{describe_dtype(original.dtype)}"
    )
    print(
        f"Input range: {np.min(original)} to {np.max(original)}"
    )

    analysis_image = make_grayscale(
        working
    )

    protected_mask = create_protected_mask(
        analysis_image
    )

    protected_percentage = (
        100.0
        * np.count_nonzero(protected_mask)
        / protected_mask.size
    )

    print(
        f"Protected image area: "
        f"{protected_percentage:.2f}%"
    )

    corrected, estimated_background = homogenise_background(
        working,
        protected_mask,
    )

    validate_preserved_pixels(
        working,
        corrected,
        protected_mask,
    )

    save_image(
        output_path,
        corrected,
        original.dtype,
        alpha,
    )

    save_diagnostics(
        output_path,
        protected_mask,
        estimated_background,
    )

    # Reopen the saved file to verify output datatype.
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        saved_check = tifffile.imread(
            output_path
        )
    else:
        saved_check = iio.imread(
            output_path
        )

    print(
        "Saved datatype and bit-depth: "
        f"{describe_dtype(saved_check.dtype)}"
    )
    print(
        f"Saved range: {np.min(saved_check)} to {np.max(saved_check)}"
    )

    if saved_check.dtype != original.dtype:
        print(
            "Warning: the saved datatype differs from the input datatype."
        )
    else:
        print(
            "Output datatype and bit-depth match the input."
        )

    print(
        f"Corrected image saved to:\n{output_path}"
    )
    print(
        "Finished."
    )


if __name__ == "__main__":
    main()
