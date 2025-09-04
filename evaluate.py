import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Separate features/target from training
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

# Load trained model
rf_model = joblib.load("model.pkl")

# ===============================
# 1. Confusion Matrix + ROC Curve
# ===============================
y_pred = rf_model.predict(X)
y_prob = rf_model.predict_proba(X)[:, 1]

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=["Predicted Real", "Predicted AI"], yticklabels=["Real", "AI"])
plt.title("Confusion Matrix")

# ROC curve
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color="darkred", lw=2, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ====================================
# 2. Logistic Regression vs RandomForest
# ====================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y)

y_prob_lr = lr_model.predict_proba(X)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr, tpr, label="RandomForest (AUC = %0.2f)" % roc_auc, color="darkred")
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = %0.2f)" % roc_auc_lr, color="darkblue")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN comparison (LogReg vs RF)")
plt.legend()
plt.show()

# ====================================
# 3. PCA feature projection
# ====================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
plt.title("PCA Feature Projection (Unsupervised)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
