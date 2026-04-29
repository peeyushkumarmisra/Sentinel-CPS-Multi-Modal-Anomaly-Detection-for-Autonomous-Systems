"""
File: cv_features.py
"""

import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern

class CVFeatureExtractor:
    @staticmethod
    def cfd(mask): # Contour Fourier Descriptors (Captures the physical outline)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(32)
        c = max(contours, key=cv2.contourArea) # Getting largest contour
        complex_contour = c[:, 0, 0] + 1j * c[:, 0, 1] # Converting to complex numbers (for Fourier Transform)
        fft_coeffs = np.fft.fft(complex_contour) # Applying fft
        mag = np.abs(fft_coeffs)
        features = mag[1:33] 
        if len(features) < 32: # Padding with zeros for smaller than 32 components
            features = np.pad(features, (0, 32 - len(features)))
        return features

    @staticmethod
    def hsv(img, mask): # Hue, Saturation, Value (Captures the color distribution)
        # Calculating a 3D histogram
        hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)],
                            [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist) # Normalizing so images
        return hist.flatten()

    @staticmethod
    def lbp(grey, mask): # Local Binary Pattern (Captures the micro-textures)
        n_pts = 8
        # Calculating LBP using 8 points and a radius of 1 pixel
        lbp_res = local_binary_pattern(grey, n_pts, 1, method='uniform')
        # Masking out the background and building a histogram of texture
        hist, _ = np.histogram(lbp_res[mask > 0], bins=n_pts + 2, range=(0, n_pts + 2))
        hist = hist.astype('float') / (hist.sum() + 1e-6)
        return hist

    # To match the output of extract() exactly
    FALLBACK = {
        'cfd': np.zeros(32),
        'hsv': np.zeros(512),
        'lbp': np.zeros(10)
        }

    def extract(self, path): # Processes a one image and returns a dict of all 4 features
        img = cv2.imread(path) if path else None
        if img is None:
            print(f"Missing OR Corrupt image → {path}")
            return self.FALLBACK.copy()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)
        return {
            'cfd': self.cfd(mask), 
            'hsv': self.hsv(img, mask), 
            'lbp': self.lbp(grey, mask)
        }

    def extract_batch(self, paths): # Gives dict of vertically stacked numpy arrays for PCA.
        results = []
        for path in tqdm(paths, desc="CV Features"):
            res = self.extract(path)
            results.append(res)
        # Stacking all into one numpy array
        batch_data = {}
        feature_keys = results[0].keys()
        for k in feature_keys:
            feature_list = [r[k] for r in results]
            batch_data[k] = np.array(feature_list)
        return batch_data