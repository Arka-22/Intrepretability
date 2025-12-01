import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# DEVICE
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(" device:", device)


# Model 

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# data preprocess

def load_numeric_data(test_size=0.2):

    ds = load_dataset("AiresPucrs/adult-census-income")["train"]
    df = ds.to_pandas().replace("?", np.nan).dropna()

    # label
    y = (df["income"] == ">50K").astype(int).values

    # correct numeric columns for THIS dataset
    numeric_cols = [
        "age",
        "fnlwgt",
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    X_raw = df[numeric_cols].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return (
        X_train, X_test,
        y_train, y_test,
        numeric_cols, scaler
    )


# Train

def train_model(model, X_train, y_train, epochs=10, batch_size=256):
    model.to(device)
    model.train()

    X = torch.tensor(X_train, dtype=torch.float32).to(device)
    y = torch.tensor(y_train, dtype=torch.long).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(epochs):
        perm = torch.randperm(len(X))
        Xs = X[perm]
        ys = y[perm]

        total_loss = 0.0
        for i in range(0, len(Xs), batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {e+1}/{epochs}, Loss = {total_loss:.4f}")



# Predict function for LIME

def make_predict_fn(model):
    def predict_fn(x):
        x = torch.tensor(np.array(x, dtype=np.float32), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        return probs
    return predict_fn



# Plot LIME

def plot_lime(exp, feature_names):
    cid = exp.available_labels()[0]
    mp = exp.as_map()[cid]
    mp_sorted = sorted(mp, key=lambda x: -abs(x[1]))

    labels = [feature_names[i] for i,_ in mp_sorted]
    values = [v for _,v in mp_sorted]

    plt.figure(figsize=(8,5))
    plt.barh(labels[::-1], values[::-1])
    plt.title("LIME Explanation (Numeric Features Only)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, numeric_cols, scaler = load_numeric_data()

    print("Numeric columns:", numeric_cols)
    print("Train shape:", X_train.shape)

    model = TwoLayerNN(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train, epochs=10)

    predict_fn = make_predict_fn(model)

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
