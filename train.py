import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Training on device:", device)


# ----------------------------
# MODEL
# ----------------------------
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ----------------------------
# LOAD + PREPROCESS DATA
# ----------------------------
def load_numeric_data(test_size=0.2):

    ds = load_dataset("AiresPucrs/adult-census-income")["train"]
    df = ds.to_pandas().replace("?", np.nan).dropna()

    y = (df["income"] == ">50K").astype(int).values
    numeric_cols = [
        "age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"
    ]
    X_raw = df[numeric_cols].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return X_train, X_test, y_train, y_test, numeric_cols, scaler


# ----------------------------
# TRAIN LOOP
# ----------------------------
def train_model(model, X_train, y_train, epochs=10, batch_size=256):
    model.to(device)
    model.train()

    X = torch.tensor(X_train, dtype=torch.float32).to(device)
    y = torch.tensor(y_train, dtype=torch.long).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        perm = torch.randperm(len(X))
        Xs = X[perm]
        ys = y[perm]

        total_loss = 0.0

        for i in range(0, len(Xs), batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]

            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss:.4f}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    X_train, X_test, y_train, y_test, numeric_cols, scaler = load_numeric_data()

    model = TwoLayerNN(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train, epochs=10)

    # SAVE MODEL
    torch.save(model.state_dict(), "model.pth")
    print("Saved model → model.pth")

    # SAVE SCALER
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved scaler → scaler.pkl")

    # SAVE feature names (recommended)
    with open("numeric_cols.pkl", "wb") as f:
        pickle.dump(numeric_cols, f)
    print("Saved feature names → numeric_cols.pkl")
