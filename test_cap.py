import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from datasets import load_dataset
from captum.attr import IntegratedGradients, NoiseTunnel
import matplotlib.pyplot as plt


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Running explanations on device:", device)


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
def load_raw_data(numeric_cols):
    ds = load_dataset("AiresPucrs/adult-census-income")["train"]
    df = ds.to_pandas().replace("?", np.nan).dropna()

    y = (df["income"] == ">50K").astype(int).values
    X_raw = df[numeric_cols].copy()

    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw.values, y, test_size=0.2, random_state=42
    )

    return X_train_raw, X_test_raw, y_train, y_test


# ----------------------------
# CAPTUM EXPLANATION
# ----------------------------
def explain_with_captum(model, X_test, feature_names, idx=10):

    model.eval()
    model.to(device)

    x = torch.tensor(X_test[idx:idx+1], dtype=torch.float32).to(device)
    x.requires_grad = True

    with torch.no_grad():
        pred_class = model(x).argmax(dim=1).item()

    print(f"\n=== Captum Explanation for index {idx} ===")
    print("Predicted class:", pred_class)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)

    attr_ig = ig.attribute(x, baselines=baseline, target=pred_class, n_steps=200)
    attr_ig = attr_ig.cpu().detach().numpy()[0]

    print("\nIntegrated Gradients:")
    for name, val in zip(feature_names, attr_ig):
        print(f"{name}: {val:+.6f}")

    nt = NoiseTunnel(ig)
    attr_sg = nt.attribute(
        x,
        baselines=baseline,
        target=pred_class,
        nt_type="smoothgrad",
        stdevs=0.1,
        nt_samples=50
    )
    attr_sg = attr_sg.cpu().detach().numpy()[0]

    print("\nSmoothGrad:")
    for name, val in zip(feature_names, attr_sg):
        print(f"{name}: {val:+.6f}")

    return attr_ig, attr_sg


# ----------------------------
# PLOT FUNCTION
# ----------------------------
def plot_attr(vals, feature_names, title):
    order = np.argsort(np.abs(vals))
    vals = np.array(vals)[order]
    names = np.array(feature_names)[order]

    plt.figure(figsize=(8,5))
    plt.barh(names, vals)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    # Load saved components
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("numeric_cols.pkl", "rb") as f:
        numeric_cols = pickle.load(f)

    # Load model
    model = TwoLayerNN(input_dim=len(numeric_cols))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    print("Loaded saved model.")

    # Load raw dataset
    X_train_raw, X_test_raw, y_train, y_test = load_raw_data(numeric_cols)

    # Scale using saved scaler
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Run explanations
    ig, sg = explain_with_captum(model, X_test, numeric_cols, idx=10)

    # Visualize
    plot_attr(ig, numeric_cols, "Integrated Gradients")
    plot_attr(sg, numeric_cols, "SmoothGrad")
