import torch
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm

from dataset import MODISDataset
from model import TrendAwareNet
from losses import total_loss

device = torch.device("cpu")
torch.set_num_threads(8)
print("Device:", device)


train_dataset = MODISDataset("./Processed", "train")
val_dataset = MODISDataset("./Processed", "val")
test_dataset = MODISDataset("./Processed", "test")
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

#loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

def evaluate(model, loader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_obs, mask, gt in tqdm(loader, desc="Validation", leave=False, ncols=120):
            x_obs = x_obs.to(device)
            mask = mask.to(device)
            gt = gt.to(device)
            pred = model(x_obs, mask)
            #print(type(total_loss))
            #print(total_loss)

            loss = total_loss(pred, gt, mask)
            total_val_loss += loss.item()
    return total_val_loss/ len(loader)


model = TrendAwareNet().to(device)   #model = TrendAwareNet().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

####early stopping
best_val_loss = float("inf")
patience = 20
counter = 0
epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=120)
    for batch_idx, (x_obs, mask, gt) in enumerate(pbar):
        x_obs = x_obs.to(device)
        mask = mask.to(device)
        gt = gt.to(device)
        pred = model(x_obs, mask)
        loss = total_loss(pred, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        avg_loss = train_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
    train_loss /= len(train_loader)
    # ===== Epoch结束后验证 =====
    val_loss = evaluate(model, val_loader)
    print(f"\nEpoch [{epoch+1:03d}/{epochs}] "
        f"Train={train_loss:.6f} "
        f"Val={val_loss:.6f}"
    )
    # ===== Save Best =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
        print(
            f"Best model saved. "
            f"Val Loss={val_loss:.6f}"
        )
    else:
        counter += 1
    # ===== Early Stop =====
    if counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
		  
print("Training finished.")
print("Best validation loss:", best_val_loss)
