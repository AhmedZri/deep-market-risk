import copy
import numpy as np
import torch
from torch import nn


class VolatilityLSTM(nn.Module):
    def __init__(self, features, hidden=16):
        super().__init__()
        self.rnn = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


class VolatilityCNN(nn.Module):
    def __init__(self, features, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(features, hidden, 3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class Autoencoder(nn.Module):
    def __init__(self, features, latent=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, 16), nn.ReLU(), nn.Linear(16, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, features)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


class RegimeLSTM(nn.Module):
    def __init__(self, features, classes=3):
        super().__init__()
        self.rnn = nn.LSTM(features, 16, batch_first=True)
        self.head = nn.Linear(16, classes)

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1])


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_model(model, x, y, vx, vy, epochs=30, classification=False, autoencoder=False):
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    dtype = torch.long if classification else torch.float32
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=dtype)
    vx = torch.as_tensor(vx, dtype=torch.float32)
    vy = torch.as_tensor(vy, dtype=dtype)
    best = float("inf")
    state = None
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for idx in order.split(64):
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(vx), vy))
        history.append({"epoch": epoch + 1, "validation_loss": val})
        if val < best - 1e-8:
            best = val
            state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    if state is None:
        raise ValueError("Nonfinite training loss")
    model.load_state_dict(state)
    model.eval()
    return history
