import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 45)
    print("    CricketGrip AI Grip Training")
    print("=" * 45)
    print()

    csv_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Clean / align rows where label may have shifted due to schema variations
    mask = df["label"].isna() & df["wrist_rotation_angle"].notna()
    if mask.any():
        df.loc[mask, "label"] = df.loc[mask, "wrist_rotation_angle"]
        df.loc[mask, "wrist_rotation_angle"] = np.nan

    # Convert numeric feature columns to float
    for col in df.columns:
        if col not in ["hand_side", "orientation", "label"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing curl values from joint angles if present
    if "index_curl" in df.columns and "index_angle" in df.columns:
        df["index_curl"] = df["index_curl"].fillna(df["index_angle"])
    if "middle_curl" in df.columns and "middle_angle" in df.columns:
        df["middle_curl"] = df["middle_curl"].fillna(df["middle_angle"])
    if "ring_curl" in df.columns and "ring_angle" in df.columns:
        df["ring_curl"] = df["ring_curl"].fillna(df["ring_angle"])

    # Fill any remaining numeric NaNs with column medians
    for col in df.columns:
        if col not in ["hand_side", "orientation", "label"]:
            df[col] = df[col].fillna(df[col].median())

    # Separate features and target
    X_raw = df.drop(columns=["label"])
    y_raw = df["label"]

    print(f"Samples: {len(df)}")
    print(f"Features: {X_raw.shape[1]}")
    print()

    # One-hot encode categorical features (hand_side, orientation)
    categorical_cols = [c for c in ["hand_side", "orientation"] if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
    X_encoded = X_encoded.astype(float)

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    print("Classes:")
    for cls_name in le.classes_:
        count = (y_raw == cls_name).sum()
        print(f"  - {cls_name} ({count} samples)")
    print()

    # Stratified Train/Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Train Random Forest Classifier
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model artifacts
    print("Saving model...")
    model_path = os.path.join(models_dir, "grip_classifier.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    columns_path = os.path.join(models_dir, "feature_columns.pkl")

    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(X_encoded.columns.tolist(), columns_path)

    print(f"  Saved: {model_path}")
    print(f"  Saved: {encoder_path}")
    print(f"  Saved: {columns_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
