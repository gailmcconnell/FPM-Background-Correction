# FPM Background Correction

**Background correction for Fourier ptychographic microscopy images while preserving specimen-level information**

## Overview

This repository provides a Python implementation of a post-reconstruction background correction method for Fourier ptychographic microscopy (FPM).

FPM enables high-resolution imaging across large fields of view, but reconstructed images can exhibit spatially varying background intensities that interfere with quantitative image analysis, particularly threshold-based segmentation and object detection.

The method reduces background non-uniformity while preserving specimen information by identifying foreground structures and selectively replacing background pixels. Unlike conventional flat-field correction approaches that modify intensities throughout an image, this method leaves pixels identified as specimen foreground unchanged during processing.

The method is applicable to both amplitude and phase reconstructions and has been evaluated using biological specimens and silica microspheres.

## Method

The correction procedure consists of four stages:

1. **Foreground detection:** The specimen is identified using local contrast, intensity gradients, and relative darkness.
2. **Foreground protection:** A binary mask is generated using morphological operations, removal of small connected components and dilation to protect specimen boundaries.
3. **Background correction:** Pixels outside the protected regions are replaced with either a constant background intensity or a smoothly varying estimated background.
4. **Output and validation:** The corrected image is saved using the original numerical data type, and protected pixels are checked for changes during processing.

Foreground detection is performed on a grayscale representation of the input image. For RGB images, correction is applied separately to each colour channel using the same foreground mask.

## Requirements

Python 3.10 or later is required.

Required packages:

- NumPy
- SciPy
- imageio
- Pillow
- tifffile

Install the dependencies using:

```bash
pip install numpy scipy imageio pillow tifffile
```

Alternatively, using Conda:

```bash
conda install -c conda-forge numpy scipy imageio pillow tifffile
```

## Usage

Download the Python script:

`FPMbackgroundcorr.py`

Open the script in Spyder, VS Code, PyCharm or another Python environment.

Under **USER SETTINGS**, specify the input and output filenames:

```python
INPUT = r"path/to/input_image.tif"
OUTPUT = r"path/to/corrected_image.tif"
```

Run the script using your preferred Python environment or from a terminal:

```bash
python FPMbackgroundcorr.py
```

The script automatically generates a foreground protection mask, performs background correction and saves the corrected image.

### Supported image formats

- TIFF (`.tif`, `.tiff`)
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

Grayscale, RGB and RGBA images are supported. Multidimensional images are not supported.


## Parameters

The following default parameters are used in the supplied implementation.

| Parameter | Default | Description |
|---|---|---|
| `DILATION_PIXELS` | 4 | Expansion of protected foreground regions |
| `LOCAL_CONTRAST_THRESHOLD` | 3.5 | Threshold for local structural contrast |
| `GRADIENT_THRESHOLD` | 4 | Threshold for intensity gradients |
| `DARKNESS_THRESHOLD` | 3 | Threshold for relatively dark structures |
| `MINIMUM_OBJECT_SIZE` | 30 | Minimum connected foreground component area, in pixels |
| `SMALL_SIGMA` | 1.5 | Small-scale Gaussian smoothing |
| `MEDIUM_SIGMA` | 8.0 | Intermediate-scale Gaussian smoothing |
| `LARGE_SIGMA` | 80.0 | Broad-scale Gaussian smoothing |
| `BACKGROUND_MODE` | `"constant"` | Background replacement mode |
| `SMOOTH_BACKGROUND_SIGMA` | 35.0 | Gaussian scale used for smooth background estimation |

The default background mode is `"constant"`, which replaces all unprotected background pixels with the median intensity of the detected background.

The alternative `"smooth"` mode replaces unprotected pixels with a smoothly varying background estimate obtained using normalised Gaussian convolution.

The parameters can be modified in the USER SETTINGS section of the script. Changes to foreground detection parameters may alter which specimen pixels are protected.

## Output files

The script produces three outputs by default:

**1. Corrected image**

The background-corrected image, saved to the specified output path.

**2. Protected foreground mask**

`*_protected_mask.png`

A binary image showing the regions identified as specimen foreground and protected from correction.

**3. Estimated background preview**

`*_estimated_background.png`

An 8-bit visualisation of the estimated replacement background.

The diagnostic images can be disabled using:

```python
SAVE_MASK = False
SAVE_BACKGROUND_ESTIMATE = False
```

## Validation and limitations

The script checks that protected foreground pixels remain unchanged during background correction, before conversion to the output data type.

For supported lossless output formats, the script retains the original numerical data type. However, processing uses 32-bit floating-point working arrays so exact numerical identity in the saved output is not guaranteed for every input data type. JPEG output is lossy and may involve conversion to 8-bit data: TIFF is recommended for quantitative analysis.

The method is designed to preserve specimen information identified by the foreground mask. Structures that are not detected as foreground may be modified during background replacement.

Users should therefore inspect the protected foreground mask to ensure that relevant specimen structures have been correctly identified.

The method is intended for post-reconstruction image correction and does not replace illumination calibration or correction during the FPM reconstruction process.


## Citation

If you use this software in published research, please cite the associated manuscript:

Copeland et al. *A method for correcting background non-uniformity while preserving specimen-level information in Fourier ptychographic microscopy images.* Biorxiv

## Licence

This software is distributed under the MIT License. See the `LICENSE` file for details.

## Author

Gail McConnell  
Centre for Biophotonics  
Strathclyde Institute of Pharmacy and Biomedical Sciences  
University of Strathclyde, Glasgow, UK
