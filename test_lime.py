import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


# DEVICE
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)


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
# LOAD RAW DATA
# ----------------------------
def load_raw_data(numeric_cols):

    ds = load_dataset("AiresPucrs/adult-census-income")["train"]
    df = ds.to_pandas().replace("?", np.nan).dropna()

    y = (df["income"] == ">50K").astype(int).values
    X_raw = df[numeric_cols].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw.values, y, test_size=0.2, random_state=42
    )

    return X_train_raw, X_test_raw, y_train, y_test


# ----------------------------
# PREDICT FUNCTION FOR LIME
# ----------------------------
def make_predict_fn(model):
    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        return probs
    return predict_fn


# ----------------------------
# LIME PLOTTING
# ----------------------------
def plot_lime(exp, feature_names):
    cid = exp.available_labels()[0]
    mp = exp.as_map()[cid]
    mp_sorted = sorted(mp, key=lambda x: -abs(x[1]))

    labels = [feature_names[i] for i,_ in mp_sorted]
    values = [v for _,v in mp_sorted]

    plt.figure(figsize=(8,5))
    plt.barh(labels[::-1], values[::-1])
    plt.title("LIME Explanation (Numeric Features)")
    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    # Load saved components
    model_state = torch.load("model.pth", map_location=device)
    scaler = pickle.load(open("scaler.pkl", "rb"))
    numeric_cols = pickle.load(open("numeric_cols.pkl", "rb"))

    # Rebuild model
    model = TwoLayerNN(input_dim=len(numeric_cols))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # Load raw data
    X_train_raw, X_test_raw, y_train, y_test = load_raw_data(numeric_cols)

    # Apply saved scaler
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    predict_fn = make_predict_fn(model)

    # LIME Explainer
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=numeric_cols,
        class_names=["<=50K", ">50K"],
        mode="classification",
        discretize_continuous=False
    )

    idx = 10
    print("True label:", y_test[idx])

    exp = explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=predict_fn,
        num_features=len(numeric_cols),
        num_samples=5000
    )

    print("\nLIME Feature Contributions:")
    for feat, val in exp.as_list():
        print(f"{feat}: {val:+.4f}")

    plot_lime(exp, numeric_cols)
