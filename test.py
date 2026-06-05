import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from dataset import MODISDataset
from model import TrendAwareNet

device = torch.device("cuda")
test_dataset = MODISDataset(
    root="./Processed",
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))
model = TrendAwareNet()
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

def moving_average(x, kernel_size=7):
    pad = kernel_size // 2
    x_pad = np.pad(x,(pad, pad), mode='edge')
    trend = np.convolve(x_pad, np.ones(kernel_size) / kernel_size, mode='valid')
    return trend

preds = []
gts = []
masks = []
obs = []
with torch.no_grad():
    for x_obs, mask, gt in test_loader:
        x_obs = x_obs.to(device)
        mask = mask.to(device)
        pred = model(x_obs, mask)
        preds.append(pred.cpu().numpy())
        gts.append(gt.numpy())
        masks.append(mask.cpu().numpy())
        obs.append(x_obs.cpu().numpy())

preds = np.concatenate(preds, axis=0)
gts = np.concatenate(gts, axis=0)
masks = np.concatenate(masks, axis=0)
obs = np.concatenate(obs, axis=0)

print("Prediction shape:", preds.shape)

missing = (1 - masks).astype(bool)
pred_missing = preds[missing]
gt_missing = gts[missing]

#####MAE
mae = mean_absolute_error(gt_missing, pred_missing)
#####RMSE
rmse = np.sqrt(mean_squared_error(gt_missing, pred_missing))
#####R2
r2 = r2_score(gt_missing, pred_missing)
#####Trend Correlation
trend_corr_list = []

for i in range(len(preds)):
    trend_pred = moving_average(preds[i])
    trend_gt = moving_average(gts[i])
    corr = np.corrcoef(trend_pred, trend_gt)[0,1]
    if not np.isnan(corr):
        trend_corr_list.append(corr)

trend_corr = np.mean(trend_corr_list)
smooth_error = np.mean(np.abs(np.diff(preds, axis=1) - np.diff(gts, axis=1)))
print("="*50)
print("RMSE  :", rmse)
print("MAE   :", mae)
print("R2    :", r2)
print("TC    :", trend_corr)
print("TSE   :", smooth_error)
print("="*50)

np.save("predictions.npy", preds)
np.save("groundtruth.npy", gts)
np.save("mask.npy", masks)
np.save("observed.npy", obs)
