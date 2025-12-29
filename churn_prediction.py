import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Robust data loader: look for CSV or Excel inside the repository's data/ folder
data_dir = Path(__file__).parent / "data"
data = None

# Candidate filenames (common variants)
candidates = [
	data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
	data_dir / "Telco-Customer-Churn.csv",
	data_dir / "Telco-Customer-Churn.xlsx",
]

for p in candidates:
	if p.exists():
		try:
			if p.suffix.lower() == ".csv":
				data = pd.read_csv(p)
			elif p.suffix.lower() in {".xls", ".xlsx"}:
				data = pd.read_excel(p)
			print(f"Loaded data from {p}")
			break
		except Exception as e:
			print(f"Found file {p} but failed to read it: {e}", file=sys.stderr)

# If no candidate matched, try finding any likely file in data_dir
if data is None and data_dir.exists():
	for p in data_dir.iterdir():
		if not p.is_file():
			continue
		name = p.name.lower()
		if "churn" in name or "telco" in name:
			try:
				if p.suffix.lower() == ".csv":
					data = pd.read_csv(p)
				elif p.suffix.lower() in {".xls", ".xlsx"}:
					data = pd.read_excel(p)
				print(f"Loaded data from {p}")
				break
			except Exception as e:
				print(f"Failed to read {p}: {e}", file=sys.stderr)

if data is None:
	files_list = list(data_dir.iterdir()) if data_dir.exists() else "(data directory missing)"
	raise FileNotFoundError(
		f"No suitable data file found in {data_dir}.\nExisting files: {files_list}\n"
		"Place the dataset as CSV or Excel inside the data/ folder."
	)

print(data.head())
print("Shape of dataset:", data.shape)
print("Columns:\n", data.columns)
print("\nDataset Info:")
# data.info() prints to stdout itself
data.info()
print("\nMissing values:\n", data.isnull().sum())
# Drop customerID column
data.drop("customerID", axis=1, inplace=True)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(inplace=True)
print("\nMissing values after cleaning:")
print(data.isnull().sum())
print("\nData types after cleaning:")
print(data.dtypes)
sns.countplot(x="Churn", data=data)
plt.title("Customer Churn Distribution")
plt.show()
sns.countplot(x="Contract", hue="Churn", data=data)
plt.title("Churn based on Contract Type")
plt.show()
sns.boxplot(x="Churn", y="tenure", data=data)
plt.title("Tenure vs Churn")
plt.show()
sns.boxplot(x="Churn", y="MonthlyCharges", data=data)
plt.title("Monthly Charges vs Churn")
plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])
print("\nEncoded data preview:")
print(data.head())
X = data.drop("Churn", axis=1)
y = data["Churn"]
X = data.drop("Churn", axis=1)
y = data["Churn"]
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optional: check first 5 rows
print("Scaled features preview:\n", X_train[:5])
from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression model
lr_model = LogisticRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on test data
y_pred_lr = lr_model.predict(X_test)

# Optional: check first 10 predictions
print("First 10 predictions:", y_pred_lr[:10])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nConfusion Matrix:\n", cm_lr)

# Optional: visualize confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on test data
y_pred_rf = rf_model.predict(X_test)

# Optional: check first 10 predictions
print("First 10 Random Forest predictions:", y_pred_rf[:10])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n", cm_rf)

# Visualize Confusion Matrix
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("Logistic Regression Accuracy:", accuracy_lr)
print("Random Forest Accuracy:", accuracy_rf)

if accuracy_rf > accuracy_lr:
    print("\nRandom Forest performs better and will be selected as the final model.")
else:
    print("\nLogistic Regression performs better and will be selected as the final model.")
import pickle

# Save the Random Forest model to a file
with open("churn_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Random Forest model saved as churn_model.pkl")
# Load the saved model (optional if already in memory)
with open("churn_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Example new customer data
# Make sure order of features matches X_train
new_customer = [[1, 0, 5, 70.35, 350.5, 1, 1, 0, 1, 0, 0, 1]]  # Replace with actual feature values

# Scale features
new_customer_scaled = scaler.transform(new_customer)

# Predict churn
prediction = loaded_model.predict(new_customer_scaled)

print("Churn Prediction (0=No, 1=Yes):", prediction[0])
