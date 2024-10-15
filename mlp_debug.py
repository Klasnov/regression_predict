import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Reg_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).squeeze()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Reg_MLP(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


torch.random.manual_seed(0)

data = pd.read_csv("data/rough_processed.csv")
X = data.drop(["price", "listing_id","indicative_price"], axis=1).to_numpy(dtype=np.float32)
y = data["price"].to_numpy(dtype=np.float32)

scaler = StandardScaler()
X_standarded = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)

train_dataset = Reg_Dataset(X_train, y_train)
val_dataset = Reg_Dataset(X_val, y_val)

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Reg_MLP(X_train.shape[1]).to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCH = 1
PATIENCE = 10
best_val_loss = float("inf")
cnt = 0

for i in range(EPOCH):
    model.train()
    train_loss = []

    for inputs, labels in train_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        del inputs, labels, outputs, loss
    
    model.eval()
    val_loss = []
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            del inputs, labels, outputs, loss

    if (i + 1) % 5 == 0:
        torch.cuda.empty_cache()
    
    avg_train_loss = np.sqrt(np.mean(train_loss))
    avg_val_loss = np.sqrt(np.mean(val_loss))

    if (i + 1) % 10 == 0:
        print(f"Epoch {i+1}/{EPOCH} -- train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f} ")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        cnt = 0
    else:
        cnt += 1
        if cnt >= PATIENCE:
            print(f"\n*****  Early Stop at Epoch {i + 1}  *****\n")
            break

model.eval()
with torch.no_grad():
    predictions = []
    groundtruth = []
    for inputs, labels in val_dataloader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs).squeeze()
        outputs = outputs.detach().cpu()
        predictions.extend(outputs.squeeze().tolist())
        groundtruth.extend(labels.squeeze().tolist())
        del inputs, labels, outputs

save_dir = "models/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), os.path.join(save_dir, "mlp_weights.pth"))
del model, criterion, optimizer
torch.cuda.empty_cache()

predictions = np.asarray(predictions)
groundtruth = np.asarray(groundtruth)

valid_rmse = np.sqrt(np.mean((predictions - groundtruth) ** 2))
print("\nMulti-layer Perceptron Regression")
print(f" - Validation RMSE: {valid_rmse:.4f}")
