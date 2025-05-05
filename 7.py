import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
data = pd.read_csv(DATA_PATH)

# Clean data
print(data.head())
print(data.info())
print(data.describe)
print(data.isnull().sum())
print(data["diagnosis"].value_counts())

data.drop(columns=["Unnamed: 32"], inplace=True)

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Features and target
x = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# PCA for 2D visualization
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# Train and evaluate Linear SVM
svc_linear = SVC(kernel='linear')
svc_linear.fit(x_train_scaled, y_train)
y_pred_linear = svc_linear.predict(x_test_scaled)
print("\n--- Linear SVM ---")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# Train and evaluate RBF SVM
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(x_train_scaled, y_train)
y_pred_rbf = svc_rbf.predict(x_test_scaled)
print("\n--- RBF SVM ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# Hyperparameter tuning
params = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto"]
}
model_search = RandomizedSearchCV(SVC(), param_distributions=params, n_iter=10, cv=5, verbose=2, n_jobs=-1)
model_search.fit(x_train_scaled, y_train)
print("\nBest Hyperparameters from RandomizedSearchCV:")
print(model_search.best_params_)

# Final prediction using best model
y_pred_best = model_search.predict(x_test_scaled)
print("\n--- Best Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# Decision Boundary Plotting
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

# Train again on PCA-reduced data for visualization
svc_vis_linear = SVC(kernel='linear')
svc_vis_linear.fit(x_train_pca, y_train)

svc_vis_rbf = SVC(kernel='rbf')
svc_vis_rbf.fit(x_train_pca, y_train)

plot_decision_boundary(x_train_pca, y_train, svc_vis_linear, "Linear SVM Decision Boundary (PCA)")
plot_decision_boundary(x_train_pca, y_train, svc_vis_rbf, "RBF SVM Decision Boundary (PCA)")