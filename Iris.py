import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ===============================
# 1️. Load Dataset
# ===============================
df = pd.read_csv("iris.csv")

print("First 5 rows of dataset:")
print(df.head())

# ===============================
# 2️. Prepare Features & Target
# ===============================
# Remove 'Id' column (not useful)
X = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

# ===============================
# 3️. Split Dataset
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4️. Train Model
# ===============================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# 5️. Make Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 6️. Accuracy
# ===============================
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# ===============================
# 7️. Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_.tolist(),
    yticklabels=model.classes_.tolist()
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Classification")
plt.show()