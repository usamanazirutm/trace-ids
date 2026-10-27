"""Differentiable detectors and their training loop."""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, h=(256, 128), p=0.2):
        super().__init__()
        layers, d = [], d_in
        for k in h:
            layers += [nn.Linear(d, k), nn.ReLU(), nn.Dropout(p)]
            d = k
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNBiLSTM(nn.Module):
    """Feature vector treated as a 1-D sequence: Conv1d x2 -> MaxPool -> BiLSTM -> [mean;max] -> MLP head."""

    def __init__(self, d_in, n_cls, c1=32, c2=64, hid=48, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, 3, padding=1), nn.ReLU(),
            nn.Conv1d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(c2, hid, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * hid, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, n_cls))

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).transpose(1, 2)  # (B, L/2, c2)
        h, _ = self.lstm(z)
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))


def build(kind, d_in, n_cls):
    return {"mlp": MLP, "cnnbilstm": CNNBiLSTM}[kind](d_in, n_cls)


def class_weights(y, n_cls):
    cnt = np.bincount(y, minlength=n_cls).astype(float)
    w = np.sqrt(cnt.max() / np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, X, bs=4096):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs])) for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def train(model, Xtr, ytr, Xva, yva, n_cls, epochs=30, bs=512, lr=1e-3, patience=5, seed=0, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, n_cls))
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    best, best_state, wait, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        pv = predict_logits(model, Xva).argmax(1)
        f1 = f1_score(yva, pv, average="macro")
        sched.step(f1)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(Xt), "val_macro_f1": f1,
                     "time_s": time.time() - t0})
        log(f"  epoch {ep + 1:02d} loss {tot / len(Xt):.4f} val macroF1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best + 1e-4:
            best, wait = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist, time.time() - t0
