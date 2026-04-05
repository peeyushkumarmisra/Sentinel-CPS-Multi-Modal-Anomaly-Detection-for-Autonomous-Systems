import os
import cv2
import torch
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from umap import UMAP
from src.cae_model import CAEmodel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from torch.utils.data import TensorDataset, DataLoader



# TRADITIONAL CV FEATURES
def cfd(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(32).tolist()
    contour = max(contours, key=cv2.contourArea) 
    comp_contour = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    fourier_ = np.fft.fft(comp_contour)
    mag = np.abs(fourier_)[1:33]
    if len(mag) < 32:
        mag = np.pad(mag, (0, 32 - len(mag)))
    return mag.tolist()

def hum(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_log.tolist()

def hsv(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()

def lbp(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    radius = 1
    n_points = 8 * radius
    n_bins = n_points + 2 
    lbp_res = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    lbp_masked = lbp_res[mask > 0]
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

def process_single_image(path):
    img = cv2.imread(path) if path else None
    if img is None:
        print(f"\nCorrupt or missing image -> {path}")
        return {'cfd': np.zeros(32).tolist(), 'hum': np.zeros(7).tolist(), 
                'hsv': np.zeros(512).tolist(), 'lbp': np.zeros(10).tolist()}
    return {
        'cfd': cfd(img),
        'hum': hum(img),
        'hsv': hsv(img),
        'lbp': lbp(img)
    }



# PROCESSING
def data_loader(df, path_map, img_size=80):
    print("Preparing Images for CAE Inference")
    images = []
    for _, row in df.iterrows():
        path = path_map.get(row['image_name'])
        img = cv2.imread(path) if path else None
        if img is None:
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        images.append(img)
    x_tensor = torch.tensor(np.array(images))
    dataset = TensorDataset(x_tensor) 
    extract_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    return extract_loader

def extract_feature_cae(model_path, extract_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAEmodel(latent_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    all_latents = []
    with torch.no_grad():
        for batch in extract_loader:
            batch_data = batch[0].to(device)
            latent_tensors = model.get_features(batch_data)
            all_latents.append(latent_tensors.cpu().numpy())
    fused_deep_features = np.vstack(all_latents)
    return fused_deep_features

def extract_feature_cv(df, path_map):
    paths = [path_map.get(name) for name in df['image_name']]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in tqdm(executor.map(process_single_image, paths), total=len(paths), desc="CV feature Extractor"):
            results.append(res)
    f_cfd = np.array([r['cfd'] for r in results])
    f_hum = np.array([r['hum'] for r in results])
    f_hsv = np.array([r['hsv'] for r in results])
    f_lbp = np.array([r['lbp'] for r in results])
    return f_cfd, f_hum, f_hsv, f_lbp



# DIMENSIONALITY REDUCTION
def reduce_dimension(features, n_components=15, use_pca=True):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    actual_components = min(n_components, scaled_features.shape[1])
    if use_pca:
        reducer = PCA(n_components=actual_components)
    else:
        reducer = UMAP(n_components=actual_components, n_jobs=-1, low_memory=False)
    return reducer.fit_transform(scaled_features)



# PIPELINE
def get_feature(base_dir, cae_model_path="models/cae_feature_ex.pt", use_pca=True):
    records = []
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png')):
                path_map[file] = os.path.join(root, file)
                stem = file.rsplit('.', 1)[0]
                parts = stem.rsplit('_', 1)
                asset_class = parts[0]
                records.append({
                    'image_name':  file,
                    'Asset Class': asset_class,
                })
    df = pd.DataFrame(records)
    print(f"\nFEATURE PIPELINE INITIATED")
    print(f"Got {len(df)} images in '{base_dir}'")

    # CAE Deep Features
    extract_loader = data_loader(df, path_map)
    cae_features = extract_feature_cae(cae_model_path, extract_loader)
    print(f"  -> CAE Features Extracted: {cae_features.shape}")
 
    # Traditional CV Features
    print(f"  -> CV Extraction for {len(df)} images")
    f_cfd, f_hum, f_hsv, f_lbp = extract_feature_cv(df, path_map)
 
    # Dimensionality Reduction
    method_name = "PCA" if use_pca else "UMAP"
    print(f"\nApplying {method_name} compression")
    fused_matrix = np.hstack((
        reduce_dimension(cae_features, use_pca=use_pca),
        reduce_dimension(f_cfd, use_pca=use_pca),
        reduce_dimension(f_hum, use_pca=use_pca),
        reduce_dimension(f_hsv, use_pca=use_pca),
        reduce_dimension(f_lbp, use_pca=use_pca)
    ))
    final_scaler = StandardScaler()
    fused_matrix_final = final_scaler.fit_transform(fused_matrix)
    df_out = df.copy()
    df_out["fused_features"] = list(fused_matrix_final)
    print(f"\nPipeline complete.")
    print(f"Final {method_name} Fused Tensor Shape: {fused_matrix_final.shape}") 
    return df_out