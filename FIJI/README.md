# FIJI/ImageJ implementation

This directory contains a Groovy implementation of the FPM background correction method for use directly within Fiji/ImageJ.

The script operates on a single reconstructed FPM amplitude or phase image. It identifies and protects specimen-containing regions before replacing the surrounding background with a uniform median intensity.

# Requirements

Fiji with Groovy scripting support

Single-plane 8-bit, 16-bit or 32-bit greyscale image

# Usage

Launch Fiji.

Select File → New → Script.

Choose Groovy as the scripting language.

Open FPM_Background_Correction.groovy in the script editor.

Click Run.

Select the reconstructed FPM image to process.

Adjust the detection thresholds and dilation parameter if required (default values are a good place to start!).

Inspect the corrected image and optional protected specimen mask.

Save the corrected image as a TIFF.

# Parameters

Dilation pixels: expands the protected specimen mask (default 4).

Local contrast threshold: identifies local specimen detail (default 3.5).

Gradient threshold: identifies specimen edges (default 4.0).

Darkness threshold: identifies locally darker specimen structures (default 3.0).

Lower thresholds generally protect more image pixels.

# Notes

This implementation is optimised for large greyscale images.

The method modifies only pixels classified as background. Users should inspect the protected specimen mask to confirm that relevant specimen features have been identified correctly.
