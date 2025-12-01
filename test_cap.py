import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients, NoiseTunnel
import matplotlib.pyplot as plt


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("device:", device)


# Model

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# data loading 

def load_numeric_data(test_size=0.2):

    ds = load_dataset("AiresPucrs/adult-census-income")["train"]
    df = ds.to_pandas().replace("?", np.nan).dropna()

    # Label
    y = (df["income"] == ">50K").astype(int).values

    # TRUE numeric columns for THIS dataset
    numeric_cols = [
        "age",
        "fnlwgt",
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    X_raw = df[numeric_cols].copy()

    # Train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return (
        X_train, X_test,
        y_train, y_test,
        numeric_cols, scaler
    )



# Training loop 

def train_model(model, X_train, y_train, epochs=20, batch_size=256):

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



# CAPTUM EXPLANATIONS

def explain_with_captum(model, X_test, feature_names, idx=10):

    model.eval()
    model.to(device)

    x = torch.tensor(X_test[idx:idx+1], dtype=torch.float32).to(device)
    x.requires_grad = True

    # predicted class
    with torch.no_grad():
        pred_class = model(x).argmax(dim=1).item()

    print(f"\n=== CAPTUM Explanation for sample {idx} ===")
    print("Predicted class:", pred_class, "\n")

    # ---------- Integrated Gradients ----------
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x).to(device)

    attr_ig = ig.attribute(
        x, baselines=baseline, target=pred_class, n_steps=200
    )
    attr_ig = attr_ig.cpu().detach().numpy()[0]

    print("Integrated Gradients:")
    for ft, val in zip(feature_names, attr_ig):
        print(f"{ft}: {val:+.6f}")

    # ---------- SmoothGrad ----------
    nt = NoiseTunnel(ig)
    attr_sg = nt.attribute(
    x,
    baselines=baseline,
    target=pred_class,
    nt_type="smoothgrad",
    stdevs=0.1,
    nt_samples=50       # <-- correct argument name
)

    attr_sg = attr_sg.cpu().detach().numpy()[0]

    print("\nSmoothGrad:")
    for ft, val in zip(feature_names, attr_sg):
        print(f"{ft}: {val:+.6f}")

    return attr_ig, attr_sg



# VISUALIZATION

def plot_attr(vals, feature_names, title):
    order = np.argsort(np.abs(vals))
    vals = np.array(vals)[order]
    names = np.array(feature_names)[order]

    plt.figure(figsize=(8,5))
    plt.barh(names, vals)
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Load data
    X_train, X_test, y_train, y_test, numeric_cols, scaler = load_numeric_data()
    print("Numeric features:", numeric_cols)

    # Train model
    model = TwoLayerNN(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train, epochs=10)

    # Captum explanations
    ig_attr, sg_attr = explain_with_captum(
        model, X_test, numeric_cols, idx=10
    )

    # Plot results
    plot_attr(ig_attr, numeric_cols, "Integrated Gradients (IG)")
    plot_attr(sg_attr, numeric_cols, "SmoothGrad (Noise-Averaged IG)")
