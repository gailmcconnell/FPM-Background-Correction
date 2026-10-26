# Fiji/ImageJ implementation

This directory contains a Java plugin (`FPM_Background_Correction.jar`) implementing the FPM background correction method for use directly within ImageJ/FIJI.

The plugin operates on a single reconstructed FPM amplitude or phase image. It identifies and protects specimen-containing regions before replacing the surrounding background with a uniform median intensity.

## Requirements

- Fiji
- Single-plane 8-bit, 16-bit or 32-bit greyscale image

## Installation

1. Download `FPM_Background_Correction.jar`.
2. Copy the JAR file into the `plugins` folder within your Fiji installation.
3. Restart Fiji.

## Usage

1. Launch Fiji.
2. Select **Plugins → FPM Background Correction**.
3. Open the reconstructed FPM image to process.
4. Adjust the detection thresholds and dilation parameter if required (default values are a good place to start!).
5. Inspect the corrected image and optional protected specimen mask.
6. Save the corrected image as a TIFF.

## Parameters

- **Dilation pixels:** expands the protected specimen mask (default 4).
- **Local contrast threshold:** identifies local specimen detail (default 3.5).
- **Gradient threshold:** identifies specimen edges (default 4.0).
- **Darkness threshold:** identifies locally darker specimen structures (default 3.0).

Lower thresholds generally protect more image pixels.

## Notes

This implementation is optimised for large greyscale images.

The method modifies only pixels classified as background. Users should inspect the protected specimen mask to confirm that relevant specimen features have been identified correctly.
