import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from datasets import load_dataset
from captum.attr import IntegratedGradients, NoiseTunnel
import matplotlib.pyplot as plt
import seaborn as sns

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
    vals = np.array(vals)
    names = np.array(feature_names)

    # Sort by absolute contribution
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    names = names[order]

    # Color map: red = negative, blue = positive
    colors = ["red" if v < 0 else "steelblue" for v in vals]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(names, vals, color=colors)

    # Add value numbers on bars
    for bar, v in zip(bars, vals):
        plt.text(
            bar.get_width() + np.sign(v)*0.01,
            bar.get_y() + bar.get_height()/2,
            f"{v:.4f}",
            va="center",
            ha="left" if v > 0 else "right"
        )

    plt.title(title, fontsize=14)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()



def plot_comparison(ig_vals, sg_vals, feature_names):
    df = pd.DataFrame({
        "feature": feature_names,
        "IG": ig_vals,
        "SG": sg_vals
    }).set_index("feature")

    df = df.iloc[np.argsort(np.abs(df["IG"]))]  # sort by IG magnitude

    plt.figure(figsize=(10,6))
    df.plot(kind="barh", figsize=(10,6))
    plt.title("Integrated Gradients vs SmoothGrad")
    plt.xlabel("Attribution")
    plt.tight_layout()
    plt.show()

def plot_heatmap(ig, sg, feature_names):
    df = pd.DataFrame({
        "IntegratedGradients": ig,
        "SmoothGrad": sg
    }, index=feature_names)

    plt.figure(figsize=(6,6))
    sns.heatmap(df, annot=True, cmap="coolwarm", center=0)
    plt.title("Attribution Heatmap (IG vs SG)")
    plt.tight_layout()
    plt.show()

def visualize_fc1_weights(model, feature_names):
    W = model.fc1.weight.detach().cpu().numpy()   # shape: (hidden_dim, input_dim)

    plt.figure(figsize=(12, 6))
    sns.heatmap(W, cmap="coolwarm", center=0,
                xticklabels=feature_names,
                yticklabels=[f"H{i}" for i in range(W.shape[0])])
    plt.title("Input → Hidden Layer (fc1) Weights")
    plt.xlabel("Input Features")
    plt.ylabel("Hidden Neurons")
    plt.tight_layout()
    plt.show()

def visualize_fc2_weights(model):
    W = model.fc2.weight.detach().cpu().numpy()  # shape (2, hidden_dim)

    plt.figure(figsize=(10, 4))
    sns.heatmap(W, cmap="coolwarm", center=0,
                xticklabels=[f"H{i}" for i in range(W.shape[1])],
                yticklabels=["class_0", "class_1"])
    plt.title("Hidden → Output Layer (fc2) Weights")
    plt.xlabel("Hidden Neurons")
    plt.ylabel("Output Classes")
    plt.tight_layout()
    plt.show()

def neuron_importance(model):
    W = model.fc1.weight.detach().cpu().numpy()
    norms = np.linalg.norm(W, axis=1)

    plt.figure(figsize=(8,4))
    plt.bar(range(len(norms)), norms)
    plt.title("Hidden Neuron Importance (L2 Norm of Input Weights)")
    plt.xlabel("Neuron")
    plt.ylabel("Weight Norm")
    plt.tight_layout()
    plt.show()

    return norms

def visualize_full_weights(model, feature_names):
    W1 = model.fc1.weight.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy()

    combined = np.vstack([W1, W2])  # stack to show all together

    plt.figure(figsize=(12, 8))
    sns.heatmap(combined, cmap="coolwarm", center=0)
    plt.title("Combined Weight Visualization (fc1 + fc2)")
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

    plot_attr(ig, numeric_cols, "Integrated Gradients")
    plot_attr(sg, numeric_cols, "SmoothGrad")
    plot_comparison(ig, sg, numeric_cols)
    plot_heatmap(ig, sg, numeric_cols)
    visualize_fc1_weights(model, numeric_cols)
    visualize_fc2_weights(model)
    neuron_importance(model)

