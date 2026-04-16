"""
File: cv_features.py

This script provides a toolkit for extracting traditional Computer Vision (CV) descriptors, 
capturing shape, texture, and color features to supplement deep learning models.
  1. CVFeatureExtractor:
    * cfd (static): Computes Contour Fourier Descriptors by converting the largest object's 
                    contour into complex numbers and performing a Fourier Transform.
    * hum (static): Calculates 7 Hu Moments (statistical shape descriptors) and applies a log 
                    transformation for scale invariance.
    * hsv (static): Generates a normalized 3D color histogram (8x8x8 bins) from the Hue, 
                    Saturation, and Value channels.
    * lbp (static): Extracts texture features using Local Binary Patterns, creating a histogram
                    of micro-patterns within the image.
    * extract: Orchestrates the individual feature extraction methods for a single image,
               providing fallback zeros if the image is corrupt.
    * extract_batch: Uses multi-threading (ThreadPoolExecutor) to process a list of images in 
                    parallel, returning the results as a dictionary of NumPy arrays.
"""

import cv2
import numpy as np
import concurrent.futures

from tqdm import tqdm
from skimage.feature import local_binary_pattern

class CVFeatureExtractor:
    """
    Extracts traditional Computer Vision descriptors from images.
    These features capture spatial, shape, color, and texture information 
    that Deep Learning models (like CAEs) sometimes overlook.
    """

    @staticmethod
    def cfd(img):
        """
        Contour Fourier Descriptors (CFD).
        Captures the closed-boundary shape of the primary object in the image.
        Useful for identifying the physical outline of industrial components.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(32)
            
        # Get the largest contour (assumed to be the main asset)
        c = max(contours, key=cv2.contourArea)
        # Convert (x,y) coordinates to complex numbers for the Fourier Transform
        complex_contour = c[:, 0, 0] + 1j * c[:, 0, 1]
        
        # Calculate the Fourier Transform and take the first 32 magnitude components
        mag = np.abs(np.fft.fft(complex_contour))[1:33]
        
        # Pad with zeros if the contour was too small to generate 32 components
        return np.pad(mag, (0, 32 - len(mag))) if len(mag) < 32 else mag

    @staticmethod
    def hum(img):
        """
        Hu Moments.
        Generates 7 statistical moments that are invariant to image scale, rotation, and reflection.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Calculate spatial moments, then Hu Moments
        hu = cv2.HuMoments(cv2.moments(mask)).flatten()
        # Log transformation handles the massive variation in scale of Hu Moments
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    @staticmethod
    def hsv(img):
        """
        HSV Color Histogram.
        Captures the color distribution of the image (Hue, Saturation, Value).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Calculate a 3D histogram with 8 bins per channel
        hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)],
                            [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        # Normalize so images of different sizes are comparable
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def lbp(img):
        """
        Local Binary Pattern (LBP).
        Captures the micro-textures of the image surface. Highly effective for 
        detecting material differences or surface wear on equipment.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        n_pts = 8
        # Calculate LBP using 8 points and a radius of 1 pixel
        lbp_res = local_binary_pattern(gray, n_pts, 1, method='uniform')
        
        # Mask out the background and build a histogram of the texture patterns
        hist, _ = np.histogram(lbp_res[mask > 0], bins=n_pts + 2, range=(0, n_pts + 2))
        hist = hist.astype('float') / (hist.sum() + 1e-6)
        return hist

    # ── Public interface ───────────────────────────────────────────────────────

    # Fallback dictionary to ensure the pipeline doesn't crash on a single bad image
    _FALLBACK = {'cfd': np.zeros(32), 'hum': np.zeros(7),
                 'hsv': np.zeros(512), 'lbp': np.zeros(10)}

    def extract(self, path):
        """Processes a single image and returns a dictionary of all 4 features."""
        img = cv2.imread(path) if path else None
        if img is None:
            print(f"  [warn] missing/corrupt image → {path}")
            return self._FALLBACK.copy()
            
        return {
            'cfd': self.cfd(img), 
            'hum': self.hum(img),
            'hsv': self.hsv(img), 
            'lbp': self.lbp(img)
        }

    def extract_batch(self, paths):
        """
        Multi-threaded batch processor.
        Returns a dictionary of vertically stacked numpy arrays ready for PCA.
        Format: {'cfd': (N, 32), 'hum': (N, 7), 'hsv': (N, 512), 'lbp': (N, 10)}
        """
        results = []
        # Utilize maximum CPU cores to process images concurrently
        with concurrent.futures.ThreadPoolExecutor() as pool:
            for res in tqdm(pool.map(self.extract, paths), total=len(paths), desc="CV Features"):
                results.append(res)
                
        # Pivot the list of dictionaries into a dictionary of arrays
        return {key: np.array([r[key] for r in results]) for key in results[0]}