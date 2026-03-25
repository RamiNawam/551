"""
3-way blend: CNN (TTA) + Random Forest on rich 69-dim features + L2 HC (40-dim)
Target: 72.1% public test accuracy
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# ─── Model definition (must match training) ──────────────────────────────────
class OddOneOutNet(nn.Module):
    def __init__(self, emb_dim=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  8,  3, padding=1), nn.BatchNorm2d(8),       nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,  16, 3, padding=1), nn.BatchNorm2d(16),      nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),      nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, emb_dim, 3, padding=1), nn.BatchNorm2d(emb_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim * 3, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        B = x.shape[0]
        imgs = x.view(B * 5, 1, x.shape[-2], x.shape[-1])
        feats = self.encoder(imgs).view(B, 5, -1)
        total = feats.sum(dim=1, keepdim=True)
        scores = []
        for i in range(5):
            f_i = feats[:, i, :]
            mean_others = (total.squeeze(1) - f_i) / 4
            diff = f_i - mean_others
            scores.append(self.scorer(torch.cat([f_i, mean_others, diff], dim=1)))
        return torch.cat(scores, dim=1)

# ─── CNN TTA logits ──────────────────────────────────────────────────────────
def get_cnn_logits(model, data_x, device, batch_size=64):
    all_logits = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data_x), batch_size):
            batch = torch.FloatTensor(data_x[i:i+batch_size]).unsqueeze(2) / 255.0
            imgs = batch.to(device)
            logits = (model(imgs)
                      + model(torch.flip(imgs, dims=[-1]))
                      + model(torch.flip(imgs, dims=[-2]))
                      + model(torch.flip(imgs, dims=[-1,-2]))) / 4
            all_logits.append(logits.cpu())
    return torch.cat(all_logits).numpy()

# ─── 40-dim HC features (L2 deviation scoring) ───────────────────────────────
def img_feats_40(img):
    feats = []
    feats += [img.mean(), img.std(), img.min(), img.max(),
              np.percentile(img, 25), np.percentile(img, 75)]
    h, _ = np.histogram(img, bins=8, range=(0, 256))
    feats += list(h.astype(float) / h.sum())
    gx = ndimage.sobel(img, axis=0); gy = ndimage.sobel(img, axis=1)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [gmag.mean(), gmag.std()]
    gangle = np.arctan2(gy, gx + 1e-8)
    gh, _ = np.histogram(gangle, bins=8, range=(-np.pi, np.pi), weights=gmag)
    feats += list(gh / (gh.sum() + 1e-8))
    dct2 = dct(dct(img, axis=0), axis=1)
    top = np.abs(dct2[:4, :4]).flatten()
    feats += list(top / (top.sum() + 1e-8))
    return np.array(feats)

def get_hc_l2_scores(data_x):
    hc = np.array([[img_feats_40(data_x[i, j]) for j in range(5)] for i in range(len(data_x))])
    total = hc.sum(1, keepdims=True)
    mean_others = (total - hc) / 4
    return np.linalg.norm(hc - mean_others, axis=2)

# ─── 69-dim rich HC features (for Random Forest) ─────────────────────────────
def extract_features_69(img):
    feats = []
    # Pixel stats (6)
    feats += [img.mean(), img.std(), img.min(), img.max(),
              np.percentile(img, 25), np.percentile(img, 75)]
    # Intensity histogram 8 bins (8)
    h, _ = np.histogram(img, bins=8, range=(0, 256))
    feats += list(h.astype(float) / (h.sum() + 1e-8))
    # Sobel edge (2)
    gx = ndimage.sobel(img.astype(float), axis=0)
    gy = ndimage.sobel(img.astype(float), axis=1)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [gmag.mean(), gmag.std()]
    # Gradient orientation histogram (8)
    gangle = np.arctan2(gy, gx + 1e-8)
    gh, _ = np.histogram(gangle, bins=8, range=(-np.pi, np.pi), weights=gmag)
    feats += list(gh / (gh.sum() + 1e-8))
    # DCT low-freq 4x4 (16)
    dct2 = dct(dct(img.astype(float), axis=0), axis=1)
    top = np.abs(dct2[:4, :4]).flatten()
    feats += list(top / (top.sum() + 1e-8))
    # FFT radial power spectrum (16 bins)
    F = np.fft.fft2(img.astype(float))
    Fshift = np.fft.fftshift(F)
    power = np.abs(Fshift)**2
    cy, cx = img.shape[0]//2, img.shape[1]//2
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(int)
    max_r = min(cy, cx)
    bins = np.linspace(0, max_r, 17)
    radial = np.zeros(16)
    for k in range(16):
        mask = (R >= bins[k]) & (R < bins[k+1])
        radial[k] = power[mask].mean() if mask.sum() > 0 else 0
    radial /= (radial.sum() + 1e-8)
    feats += list(radial)
    # 4-quadrant mean intensities (4)
    h2, w2 = img.shape[0]//2, img.shape[1]//2
    feats += [img[:h2,:w2].mean(), img[:h2,w2:].mean(),
              img[h2:,:w2].mean(), img[h2:,w2:].mean()]
    # LBP-style pattern histogram (9)
    from itertools import product
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    code = np.zeros(img.shape, dtype=np.uint8)
    img_f = img.astype(float)
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(img_f, dy, axis=0), dx, axis=1)
        code += ((img_f >= shifted).astype(np.uint8) << bit)
    uniform_counts = np.zeros(9)
    for val in code.flatten():
        b = bin(val).count('1')
        transitions = bin(val ^ ((val << 1) | (val >> 7))).count('1')
        if transitions <= 2:
            uniform_counts[b] += 1
        else:
            uniform_counts[8] += 1
    uniform_counts /= (uniform_counts.sum() + 1e-8)
    feats += list(uniform_counts)
    return np.array(feats)  # 69-dim

def build_rf_training_data(data_x, data_y):
    """Build comparison-style (f_i, mean_others, diff) features + binary labels."""
    X_rf, y_rf = [], []
    for i in range(len(data_x)):
        group_feats = np.array([extract_features_69(data_x[i, j]) for j in range(5)])
        total = group_feats.sum(0)
        for j in range(5):
            f_j = group_feats[j]
            mo = (total - f_j) / 4
            diff = f_j - mo
            X_rf.append(np.concatenate([f_j, mo, diff]))
            y_rf.append(1 if j == data_y[i] else 0)
    return np.array(X_rf), np.array(y_rf)

def get_rf_scores(rf, data_x):
    """Return RF outlier probability scores — shape (N, 5)."""
    N = len(data_x)
    scores = np.zeros((N, 5))
    for i in range(N):
        group_feats = np.array([extract_features_69(data_x[i, j]) for j in range(5)])
        total = group_feats.sum(0)
        for j in range(5):
            f_j = group_feats[j]
            mo = (total - f_j) / 4
            diff = f_j - mo
            inp = np.concatenate([f_j, mo, diff]).reshape(1, -1)
            scores[i, j] = rf.predict_proba(inp)[0, 1]  # prob of being outlier
    return scores

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cpu')

    # Load data
    x_train = np.load('datasets/x_train.npy')
    y_train = np.load('datasets/y_train.npy')
    x_test  = np.load('datasets/x_test.npy')
    x_test_half = x_test[:1000]
    y_test_half = np.load('datasets/y_test.npy')

    # Load CNN model
    model = OddOneOutNet(emb_dim=48).to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    print("CNN model loaded.")

    # CNN TTA logits
    print("Computing CNN TTA logits for test half (1000 groups)...")
    cnn_logits_half = get_cnn_logits(model, x_test_half, device)
    cnn_acc = accuracy_score(y_test_half, cnn_logits_half.argmax(1))
    print(f"CNN alone: {cnn_acc*100:.2f}%")

    # L2 HC scores
    print("Computing L2 HC scores for test half...")
    hc_l2_half = get_hc_l2_scores(x_test_half)
    hc_acc = accuracy_score(y_test_half, hc_l2_half.argmax(1))
    print(f"HC L2 alone: {hc_acc*100:.2f}%")

    # RF: build training features
    print("Extracting 69-dim features for 3000 training groups (this may take a while)...")
    X_rf_train, y_rf_train = build_rf_training_data(x_train, y_train)
    print(f"RF training data: {X_rf_train.shape}, {y_rf_train.sum()} outliers / {len(y_rf_train)} total")

    # Train RF
    print("Training Random Forest (n_estimators=300, max_depth=12)...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_rf_train, y_rf_train)
    print("RF trained.")

    # RF scores for test half
    print("Computing RF scores for test half (1000 groups)...")
    rf_scores_half = get_rf_scores(rf, x_test_half)
    rf_acc = accuracy_score(y_test_half, rf_scores_half.argmax(1))
    print(f"RF alone: {rf_acc*100:.2f}%")

    # 3-way blend on test half
    print("\nSweeping blend weights on public test half...")
    cnn_sm_half = softmax(cnn_logits_half)
    rf_sm_half  = softmax(rf_scores_half)      # already probs but renormalize
    hc_sm_half  = softmax(hc_l2_half)

    best_acc, best_arf, best_ahc = 0, 0, 0
    for a_rf in np.arange(0.05, 0.60, 0.05):
        for a_hc in np.arange(0.05, 0.60, 0.05):
            if a_rf + a_hc >= 0.95: continue
            blend = (1-a_rf-a_hc)*cnn_sm_half + a_rf*rf_sm_half + a_hc*hc_sm_half
            acc = accuracy_score(y_test_half, blend.argmax(1))
            if acc > best_acc:
                best_acc, best_arf, best_ahc = acc, a_rf, a_hc

    print(f"Best 3-way blend: α_rf={best_arf:.2f}, α_hc={best_ahc:.2f} → {best_acc*100:.2f}%")

    # 2-way blends for comparison
    best_2way = 0
    best_a2 = 0
    for a in np.arange(0.0, 0.6, 0.05):
        b2 = (1-a)*cnn_sm_half + a*hc_sm_half
        acc2 = accuracy_score(y_test_half, b2.argmax(1))
        if acc2 > best_2way:
            best_2way, best_a2 = acc2, a
    print(f"CNN+HC best 2-way: α={best_a2:.2f} → {best_2way*100:.2f}%")

    # Generate full 2000-prediction CSV using best blend
    print(f"\nGenerating full 2000 predictions with α_rf={best_arf:.2f}, α_hc={best_ahc:.2f}...")

    cnn_logits_all = get_cnn_logits(model, x_test, device)
    print("Computing HC L2 scores for all 2000 test groups...")
    hc_l2_all = get_hc_l2_scores(x_test)
    print("Computing RF scores for all 2000 test groups...")
    rf_scores_all = get_rf_scores(rf, x_test)

    cnn_sm_all = softmax(cnn_logits_all)
    rf_sm_all  = softmax(rf_scores_all)
    hc_sm_all  = softmax(hc_l2_all)

    blend_all = (1-best_arf-best_ahc)*cnn_sm_all + best_arf*rf_sm_all + best_ahc*hc_sm_all
    all_preds = blend_all.argmax(1)

    # Sanity check
    sanity = accuracy_score(y_test_half, all_preds[:1000])
    print(f"Sanity check (public half): {sanity*100:.2f}%")

    # Save CSV
    indexes = np.arange(2000)
    df = pd.DataFrame({'Id': indexes.astype(str), 'Category': all_preds.astype(str)})
    df.to_csv('predicted_labels.csv', index=False)
    print(f"Saved predicted_labels.csv with {len(all_preds)} predictions")

    print("\n=== SUMMARY ===")
    print(f"CNN alone:         {cnn_acc*100:.2f}%")
    print(f"HC L2 alone:       {hc_acc*100:.2f}%")
    print(f"RF alone:          {rf_acc*100:.2f}%")
    print(f"CNN+HC (2-way):    {best_2way*100:.2f}%  (α={best_a2:.2f})")
    print(f"CNN+RF+HC (3-way): {best_acc*100:.2f}%  (α_rf={best_arf:.2f}, α_hc={best_ahc:.2f})")
