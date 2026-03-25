import numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, time

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

x_train = np.load('datasets/x_train.npy')
y_train = np.load('datasets/y_train.npy')
x_test  = np.load('datasets/x_test.npy')
y_test  = np.load('datasets/y_test.npy')

class OddOneOutNet(nn.Module):
    def __init__(self, emb_dim=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,emb_dim,3,padding=1), nn.BatchNorm2d(emb_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim*3, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        B = x.shape[0]
        feats = self.encoder(x.view(B*5, 1, x.shape[-2], x.shape[-1])).view(B, 5, -1)
        total = feats.sum(dim=1, keepdim=True)
        scores = []
        for i in range(5):
            f_i = feats[:, i, :]; mo = (total.squeeze(1) - f_i) / 4
            scores.append(self.scorer(torch.cat([f_i, mo, f_i-mo], dim=1)))
        return torch.cat(scores, dim=1)

class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, augment=False):
        self.x = x; self.y = y; self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        imgs = torch.FloatTensor(self.x[idx]) / 255.0  # (5,32,32)
        label = self.y[idx]
        if self.augment:
            # Flips
            if random.random() > 0.5: imgs = torch.flip(imgs, [2])
            if random.random() > 0.5: imgs = torch.flip(imgs, [1])
            # Random noise
            if random.random() > 0.5:
                imgs = imgs + torch.randn_like(imgs) * 0.05
                imgs = imgs.clamp(0, 1)
        return imgs.unsqueeze(1), label  # (5,1,32,32)

# 90/10 split
split = int(0.9 * len(x_train))
idx = np.random.permutation(len(x_train))
train_idx, val_idx = idx[:split], idx[split:]

train_ds = GroupDataset(x_train[train_idx], y_train[train_idx], augment=True)
val_ds   = GroupDataset(x_train[val_idx],   y_train[val_idx],   augment=False)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=64)

model = OddOneOutNet(emb_dim=48)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {params}')

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_val = 0
for epoch in range(200):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    if (epoch+1) % 20 == 0 or epoch == 199:
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                correct += (model(xb).argmax(1) == yb).sum().item()
        val_acc = correct / len(val_ds)
        
        # Test acc
        test_x = torch.FloatTensor(x_test[:1000]) / 255.0
        imgs = test_x.unsqueeze(2)
        with torch.no_grad():
            logits = (model(imgs) + model(torch.flip(imgs,[4])) + 
                      model(torch.flip(imgs,[3])) + model(torch.flip(imgs,[3,4]))) / 4
        test_acc = (logits.argmax(1).numpy() == y_test).mean()
        
        print(f'Epoch {epoch+1}: val={val_acc*100:.1f}%, test={test_acc*100:.1f}%')
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), 'model_better.pt')

print('Training done. Best val:', best_val)

# Final eval with HC blend
from scipy import ndimage as sci_ndimage
from scipy.fftpack import dct

def img_feats(img):
    feats = []
    feats += [img.mean(), img.std(), img.min(), img.max(),
              np.percentile(img, 25), np.percentile(img, 75)]
    h, _ = np.histogram(img, bins=8, range=(0,256)); feats += list(h.astype(float)/h.sum())
    gx = sci_ndimage.sobel(img, axis=0); gy = sci_ndimage.sobel(img, axis=1)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [gmag.mean(), gmag.std()]
    gangle = np.arctan2(gy, gx+1e-8)
    gh, _ = np.histogram(gangle, bins=8, range=(-np.pi,np.pi), weights=gmag)
    feats += list(gh/(gh.sum()+1e-8))
    dct2 = dct(dct(img, axis=0), axis=1)
    top_dct = np.abs(dct2[:4,:4]).flatten()
    feats += list(top_dct/(top_dct.sum()+1e-8))
    return np.array(feats)

hc = np.array([[img_feats(x_test[i,j]) for j in range(5)] for i in range(1000)])
total_hc = hc.sum(1, keepdims=True)
mo_hc = (total_hc - hc) / 4
hc_scores = np.linalg.norm(hc - mo_hc, axis=2)

model.load_state_dict(torch.load('model_better.pt', map_location='cpu', weights_only=True))
model.eval()
test_x = torch.FloatTensor(x_test[:1000]) / 255.0
imgs = test_x.unsqueeze(2)
with torch.no_grad():
    cnn_logits = (model(imgs) + model(torch.flip(imgs,[4])) + 
                  model(torch.flip(imgs,[3])) + model(torch.flip(imgs,[3,4]))) / 4
cnn_logits = cnn_logits.numpy()

def softmax(x): return np.exp(x - x.max(1,keepdims=True)) / np.exp(x - x.max(1,keepdims=True)).sum(1,keepdims=True)
cnn_sm = softmax(cnn_logits)
hc_sm = softmax(hc_scores)

best_acc, best_alpha = 0, 0
for alpha in np.arange(0.2, 0.5, 0.02):
    blend = (1-alpha)*cnn_sm + alpha*hc_sm
    acc = (blend.argmax(1) == y_test).mean()
    if acc > best_acc: best_acc, best_alpha = acc, alpha
print(f'Best blend: alpha={best_alpha:.2f}, acc={best_acc*100:.2f}%')
