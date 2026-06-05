import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm


ROOT = r".\MOD13Q1\image"
SAVE_DIR = r".\MOD13Q1\Processed"
os.makedirs(SAVE_DIR, exist_ok=True)


ndvi_files = sorted(glob.glob(os.path.join(ROOT, "*NDVI*.tif")))
qa_files = sorted(glob.glob(os.path.join(ROOT, "*VI_Quality*.tif")))
print("NDVI:", len(ndvi_files))
print("QA  :", len(qa_files))
assert len(ndvi_files) == len(qa_files)

# ==================================================
# READ DATA
# ==================================================

ndvi_cube = []
mask_cube = []
for ndvi_path, qa_path in tqdm(zip(ndvi_files, qa_files), total=len(ndvi_files)):
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1).astype(np.float32)
    with rasterio.open(qa_path) as src:
        qa = src.read(1)
    # MODIS scale factor
    ndvi = ndvi * 0.0001

    #mask = (qa <= 1).astype(np.uint8)
    quality = qa & 0b11
    mask = np.where(quality <= 1, 1, 0).astype(np.uint8)
    #mask = (quality <= 1)
    #print("mask:",mask.mean())
	
    ndvi_cube.append(ndvi)
    mask_cube.append(mask)
ndvi_cube = np.stack(ndvi_cube)
mask_cube = np.stack(mask_cube)
print("Cube shape:", ndvi_cube.shape)

T,H,W = ndvi_cube.shape
X = ndvi_cube.reshape(T,-1).T
Mask = mask_cube.reshape(T,-1).T
print("Raw samples:", X.shape)

# ==================================================
# REMOVE INVALID PIXELS
# ==================================================

valid = np.ones(len(X), dtype=bool)

valid &= (Mask.sum(axis=1) > 50)

valid &= (X.max(axis=1) < 1.0)
valid &= (X.min(axis=1) > -0.2)

X = X[valid]
Mask = Mask[valid]
print("Valid samples:", X.shape)

X_obs = X.copy()
X_obs[Mask == 0] = 0

np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.pah.join(SAVE_DIR, "Mask.npy"), Mask)
np.save(os.path.join(SAVE_DIR, "X_obs.npy"), X_obs)
print("Saved")

print("X     :", X.shape)
print("Mask  :", Mask.shape)
print("X_obs :", X_obs.shape)

print(X.shape)
print(Mask.mean())
print(Mask.sum(axis=1).min())
print(Mask.sum(axis=1).max())
