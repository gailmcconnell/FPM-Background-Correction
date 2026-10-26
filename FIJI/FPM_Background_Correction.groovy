/*
 * FPM Background Correction for Fiji/ImageJ
 * Gail McConnell, University of Strathclyde, 4 August 2026
 * Designed for large single-plane 8-bit, 16-bit and 32-bit grayscale images.
 */

import ij.IJ
import ij.ImagePlus
import ij.gui.GenericDialog
import ij.io.OpenDialog
import ij.io.SaveDialog
import ij.plugin.filter.GaussianBlur
import ij.process.*
import java.util.Arrays

final int DEFAULT_DILATION = 4
final double DEFAULT_LOCAL_CONTRAST = 3.5
final double DEFAULT_GRADIENT = 4.0
final double DEFAULT_DARKNESS = 3.0
final double SMALL_SIGMA = 1.5
final double MEDIUM_SIGMA = 8.0
final double LARGE_SIGMA = 80.0
final int MAX_STAT_SAMPLES = 1000000

// Choose input.
def od = new OpenDialog("Select image to correct", null)
String inputPath = od.getPath()
if (inputPath == null) return

ImagePlus imp = IJ.openImage(inputPath)
if (imp == null) {
    IJ.error("Could not open the selected image.")
    return
}
if (imp.getStackSize() != 1 || imp.getNChannels() != 1 || imp.getBitDepth() == 24) {
    IJ.error("This version supports one grayscale image plane (8-, 16- or 32-bit).")
    imp.close()
    return
}

// Editable parameters.
def gd = new GenericDialog("FPM Background Correction")
gd.addNumericField("Dilation pixels", DEFAULT_DILATION, 0)
gd.addNumericField("Local contrast threshold", DEFAULT_LOCAL_CONTRAST, 2)
gd.addNumericField("Gradient threshold", DEFAULT_GRADIENT, 2)
gd.addNumericField("Darkness threshold", DEFAULT_DARKNESS, 2)
gd.addCheckbox("Show protected mask", true)
gd.addMessage("Lower thresholds protect more pixels")
gd.showDialog()
if (gd.wasCanceled()) { imp.close(); return }

int dilationPixels = Math.max(0, (int)Math.round(gd.getNextNumber()))
double localThreshold = gd.getNextNumber()
double gradientThreshold = gd.getNextNumber()
double darknessThreshold = gd.getNextNumber()
boolean showMask = gd.getNextBoolean()

long started = System.currentTimeMillis()
int bitDepth = imp.getBitDepth()
ImageProcessor original = imp.getProcessor().duplicate()
FloatProcessor sourceFP = (FloatProcessor) imp.getProcessor().convertToFloatProcessor()
int w = sourceFP.getWidth(), h = sourceFP.getHeight(), n = w*h
float[] src = ((float[])sourceFP.getPixels()).clone()
byte[] maskPixels = new byte[n]

IJ.showStatus("Gaussian filtering: fine scale...")
IJ.showProgress(0.05)
FloatProcessor fineFP = new FloatProcessor(w, h, src.clone())
new GaussianBlur().blurGaussian(fineFP, SMALL_SIGMA, SMALL_SIGMA, 0.01)
float[] fine = (float[])fineFP.getPixels()

IJ.showStatus("Gaussian filtering: medium scale...")
IJ.showProgress(0.15)
FloatProcessor mediumFP = new FloatProcessor(w, h, src.clone())
new GaussianBlur().blurGaussian(mediumFP, MEDIUM_SIGMA, MEDIUM_SIGMA, 0.01)
float[] medium = (float[])mediumFP.getPixels()

// Local contrast: estimate median/MAD from a sample, then threshold full image.
IJ.showStatus("Detecting local contrast...")
IJ.showProgress(0.28)
def detailStats = sampledStatsDifference(fine, medium, true, MAX_STAT_SAMPLES)
double detailCut = detailStats[0] + localThreshold * detailStats[1]
for (int i=0; i<n; i++) {
    if (Math.abs(fine[i]-medium[i]) > detailCut) maskPixels[i] = (byte)255
}
medium = null; mediumFP = null
System.gc()

IJ.showStatus("Gaussian filtering: broad scale...")
IJ.showProgress(0.40)
FloatProcessor broadFP = new FloatProcessor(w, h, src.clone())
new GaussianBlur().blurGaussian(broadFP, LARGE_SIGMA, LARGE_SIGMA, 0.01)
float[] broad = (float[])broadFP.getPixels()

IJ.showStatus("Detecting dark structures...")
IJ.showProgress(0.57)
def darkStats = sampledStatsDifference(broad, fine, false, MAX_STAT_SAMPLES)
double darkCut = darkStats[0] + darknessThreshold * darkStats[1]
for (int i=0; i<n; i++) {
    if ((broad[i]-fine[i]) > darkCut) maskPixels[i] = (byte)255
}
broad = null; broadFP = null
System.gc()

// Sobel using ImageJ's compiled convolution code.
IJ.showStatus("Calculating Sobel gradient...")
IJ.showProgress(0.66)
FloatProcessor gxFP = (FloatProcessor)fineFP.duplicate()
FloatProcessor gyFP = (FloatProcessor)fineFP.duplicate()
float[] kx = [-1f,0f,1f,-2f,0f,2f,-1f,0f,1f] as float[]
float[] ky = [-1f,-2f,-1f,0f,0f,0f,1f,2f,1f] as float[]
gxFP.convolve(kx,3,3)
gyFP.convolve(ky,3,3)
float[] gx = (float[])gxFP.getPixels()
float[] gy = (float[])gyFP.getPixels()
def gradStats = sampledGradientStats(gx, gy, MAX_STAT_SAMPLES)
double gradCut = gradStats[0] + gradientThreshold * gradStats[1]
for (int i=0; i<n; i++) {
    if (Math.hypot(gx[i],gy[i]) > gradCut) maskPixels[i] = (byte)255
}
fine = null; fineFP = null; gx = null; gy = null; gxFP = null; gyFP = null
System.gc()

// Fast compiled binary morphology.
IJ.showStatus("Cleaning and expanding mask...")
IJ.showProgress(0.80)
ByteProcessor maskBP = new ByteProcessor(w,h,maskPixels,null)
maskBP.setBinaryThreshold()
BinaryProcessor bp = new BinaryProcessor(maskBP)
bp.dilate(); bp.dilate(); bp.erode(); bp.erode()  // closing, two iterations
fillHolesNative(bp)
for (int d=0; d<dilationPixels; d++) bp.dilate()
maskPixels = (byte[])bp.getPixels()

// Exact background median for integer images via histogram; sampled for float.
IJ.showStatus("Calculating background value...")
IJ.showProgress(0.88)
double backgroundTarget
if (bitDepth == 8 || bitDepth == 16) {
    backgroundTarget = maskedIntegerMedian(original, maskPixels)
} else {
    backgroundTarget = maskedSampleMedian(src, maskPixels, MAX_STAT_SAMPLES)
}
if (Double.isNaN(backgroundTarget)) {
    IJ.error("No unprotected background pixels remain. Raise thresholds or reduce dilation.")
    imp.close(); return
}

IJ.showStatus("Creating corrected image...")
IJ.showProgress(0.94)
float[] corrected = src
for (int i=0; i<n; i++) if ((maskPixels[i] & 0xff)==0) corrected[i]=(float)backgroundTarget
FloatProcessor correctedFP = new FloatProcessor(w,h,corrected)
ImageProcessor outIP = bitDepth==8 ? correctedFP.convertToByte(false) :
                       bitDepth==16 ? correctedFP.convertToShort(false) : correctedFP
String base = imp.getTitle().replaceFirst(/(?i)\.(tif|tiff|png|jpg|jpeg)$/, "")
ImagePlus result = new ImagePlus(base+"_background_corrected",outIP)
result.setCalibration(imp.getCalibration().copy())
result.show()
if (showMask) new ImagePlus(base+"_protected_mask",bp).show()

IJ.showProgress(1.0)
long seconds = Math.round((System.currentTimeMillis()-started)/1000.0)
IJ.log("FPM correction completed in " + seconds + " s")
IJ.log("Background target: " + backgroundTarget)

// Save dialog.
def sd = new SaveDialog("Save corrected image",base+"_background_corrected",".tif")
if (sd.getFileName()!=null) {
    String outputPath = sd.getDirectory()+sd.getFileName()
    IJ.saveAsTiff(result,outputPath)
    IJ.showStatus("Saved: "+outputPath)
} else IJ.showStatus("Processing complete; image not saved.")
imp.close()

// ---------------- functions ----------------

double[] sampledStatsDifference(float[] a, float[] b, boolean absoluteValue, int maxSamples) {
    int n=a.length
    int step=Math.max(1,(int)Math.ceil(n/(double)maxSamples))
    int count=(int)Math.ceil(n/(double)step)
    float[] sample=new float[count]
    int j=0
    for (int i=0;i<n;i+=step) {
        float v=a[i]-b[i]
        sample[j++]=absoluteValue ? Math.abs(v) : v
    }
    if (j<sample.length) sample=Arrays.copyOf(sample,j)
    return medianMad(sample)
}

double[] sampledGradientStats(float[] gx,float[] gy,int maxSamples) {
    int n=gx.length
    int step=Math.max(1,(int)Math.ceil(n/(double)maxSamples))
    int count=(int)Math.ceil(n/(double)step)
    float[] sample=new float[count]
    int j=0
    for (int i=0;i<n;i+=step) sample[j++]=(float)Math.hypot(gx[i],gy[i])
    if (j<sample.length) sample=Arrays.copyOf(sample,j)
    return medianMad(sample)
}

double[] medianMad(float[] values) {
    Arrays.sort(values)
    double med=arrayMedianSorted(values)
    float[] dev=new float[values.length]
    for (int i=0;i<values.length;i++) dev[i]=(float)Math.abs(values[i]-med)
    Arrays.sort(dev)
    double mad=arrayMedianSorted(dev)
    double sigma=1.4826*mad
    if (sigma<1e-7) sigma=1.0
    return [med,sigma] as double[]
}

double arrayMedianSorted(float[] v) {
    int m=v.length/2
    return (v.length%2==1) ? v[m] : 0.5*(v[m-1]+v[m])
}

void fillHolesNative(BinaryProcessor bp) {
    // Flood-fill the exterior of the inverted mask, then invert back.
    int w=bp.getWidth(),h=bp.getHeight()
    ByteProcessor inv=(ByteProcessor)bp.duplicate()
    inv.invert()
    FloodFiller ff=new FloodFiller(inv)
    inv.setValue(0)
    for (int x=0;x<w;x++) {
        if (inv.getPixel(x,0)==255) ff.fill(x,0)
        if (inv.getPixel(x,h-1)==255) ff.fill(x,h-1)
    }
    for (int y=0;y<h;y++) {
        if (inv.getPixel(0,y)==255) ff.fill(0,y)
        if (inv.getPixel(w-1,y)==255) ff.fill(w-1,y)
    }
    byte[] p=(byte[])bp.getPixels()
    byte[] q=(byte[])inv.getPixels()
    for (int i=0;i<p.length;i++) if ((q[i]&0xff)==255) p[i]=(byte)255
}

double maskedIntegerMedian(ImageProcessor ip, byte[] mask) {
    int bins=ip.getBitDepth()==8 ? 256 : 65536
    long[] hist=new long[bins]
    long total=0
    for (int i=0;i<mask.length;i++) if ((mask[i]&0xff)==0) {
        int v=(int)ip.getf(i)
        if (v<0) v=0; else if (v>=bins) v=bins-1
        hist[v]++; total++
    }
    if (total==0) return Double.NaN
    long target=(total-1)/2, cumulative=0
    for (int i=0;i<bins;i++) { cumulative+=hist[i]; if (cumulative>target) return i }
    return Double.NaN
}

double maskedSampleMedian(float[] src,byte[] mask,int maxSamples) {
    int n=src.length
    int step=Math.max(1,(int)Math.ceil(n/(double)maxSamples))
    float[] sample=new float[Math.min(maxSamples,(int)Math.ceil(n/(double)step))]
    int j=0
    for (int i=0;i<n && j<sample.length;i+=step) if ((mask[i]&0xff)==0) sample[j++]=src[i]
    if (j==0) return Double.NaN
    sample=Arrays.copyOf(sample,j); Arrays.sort(sample)
    return arrayMedianSorted(sample)
}
