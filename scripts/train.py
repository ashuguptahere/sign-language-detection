import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("hand_keypoints.csv")

# Load data (Feature, Labels) for training
X = df.drop("label", axis=1)
y = df["label"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = svm.predict(X_test_scaled)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Saving the model
os.makedirs("models/", exist_ok=True)
joblib.dump(svm, "models/svm_hand_sign_model.pkl")
joblib.dump(list(svm.classes_), "models/labels.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Trained SVM model, labels and scaler are saved as 'models/svm_hand_sign_model.pkl', 'models/labels.pkl' and 'models/scaler.pkl'")
