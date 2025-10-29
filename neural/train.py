# neural/train.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from core.geometry import random_triangles, triangle_centroids
from core.metrics import compute_sah_for_split
from neural.model import SimpleSplitNet

# ------------------------------
# Dataset sintético de treinamento
# ------------------------------
class SplitDataset(Dataset):
    def __init__(self, n_samples=500, tris_per_sample=64):
        self.data = []
        for _ in range(n_samples):
            tris = random_triangles(tris_per_sample, scale=1.0, seed=np.random.randint(1e6))
            cents = triangle_centroids(tris)
            feat = np.concatenate([cents.mean(axis=0), cents.std(axis=0)])  # 6 features
            axis = int(np.argmax((tris.reshape(-1, 3).max(axis=0) - tris.reshape(-1, 3).min(axis=0))))
            cand_positions = np.unique(np.percentile(cents[:, axis], [5, 25, 50, 75, 95]))
            best_pos, best_cost = 0.5, np.inf
            for pos in cand_positions:
                cost, l, r = compute_sah_for_split(list(range(len(tris))), tris, axis, pos)
                if cost < best_cost and len(l) > 0 and len(r) > 0:
                    best_cost = cost
                    best_pos = pos
            mins, maxs = cents[:, axis].min(), cents[:, axis].max()
            norm_pos = (best_pos - mins) / (maxs - mins + 1e-9)
            self.data.append((feat.astype(np.float32), np.float32(norm_pos)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f, t = self.data[idx]
        return f, t


# ------------------------------
# Treinamento por época
# ------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        # garante que dados e rótulos estão no mesmo device do modelo
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


# ------------------------------
# Função principal de treinamento
# ------------------------------
def run_training(epochs=5):
    # 1️⃣ Detecta automaticamente o dispositivo
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _ = torch.tensor([0.0]).to(device)
            print(f"✅ Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError("CUDA não disponível")
    except Exception as e:
        print(f"⚠️ Falha ao usar CUDA: {e}")
        device = torch.device("cpu")
        print("➡️ Treinando em CPU.")

    # 2️⃣ Cria dataset e DataLoader
    ds = SplitDataset(n_samples=300, tris_per_sample=64)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    # 3️⃣ Cria o modelo e envia para o mesmo device
    model = SimpleSplitNet(input_dim=6).to(device)

    # 4️⃣ Otimizador
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5️⃣ Loop de treinamento
    for e in range(epochs):
        loss = train_epoch(model, dl, optimizer, device)
        print(f"Epoch {e+1}/{epochs}  Loss={loss:.6f}")

    print("🏁 Treinamento finalizado.")
    return model
