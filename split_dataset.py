import numpy as np
from sklearn.model_selection import train_test_split

X = np.load(r".\Processed/X.npy")
Mask = np.load(r".\Processed/Mask.npy")
X_obs = np.load(r".\MOD13Q1\Processed/X_obs.npy")

idx = np.arange(len(X))

train_idx, test_idx = train_test_split(
    idx,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.1,
    random_state=42
)

np.save(r".\Processed/train_idx.npy", train_idx)
np.save(r".\MOD13Q1\Processed/val_idx.npy", val_idx)
np.save(r".\MOD13Q1\Processed/test_idx.npy", test_idx)

print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))
