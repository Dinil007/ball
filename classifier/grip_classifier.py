import os
import joblib
import numpy as np
import pandas as pd


class GripClassifier:
    """
    Cricket grip classifier that uses trained Random Forest, LabelEncoder,
    and 24-feature schema to predict grip types in real-time.
    """

    def __init__(self, models_dir="models", confidence_threshold=0.60):
        """
        Load trained model artifacts from the specified models directory.

        Parameters:
            models_dir (str): Path to models folder.
            confidence_threshold (float): Minimum confidence required (default 0.60 / 60%).
        """
        self.models_dir = models_dir
        self.confidence_threshold = float(confidence_threshold)
        self.model_path = os.path.join(models_dir, "grip_classifier.pkl")
        self.encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        self.features_path = os.path.join(models_dir, "feature_columns.pkl")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {self.encoder_path}")
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Feature columns file not found: {self.features_path}")

        # Load model artifacts
        self.model = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)
        self.feature_columns = joblib.load(self.features_path)

    def verify_alignment(self, df_encoded):
        """
        Diagnostic helper to check feature alignment between runtime dataframe
        and model expected schema.
        """
        received_cols = set(df_encoded.columns)
        expected_cols = set(self.feature_columns)
        missing_cols = expected_cols - received_cols
        extra_cols = received_cols - expected_cols

        return {
            "features_received": len(received_cols),
            "features_expected": len(expected_cols),
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols)
        }

    def predict(self, features):
        """
        Predict cricket grip label and confidence from extracted kinematic features.
        Only classifies when orientation is BACK.

        Parameters:
            features (dict): Feature dictionary from FeatureExtractor.

        Returns:
            dict: {
                "grip": str,
                "raw_grip": str,
                "confidence": float,
                "probabilities": dict,
                "alignment": dict
            }
        """
        if not features:
            return {
                "grip": "NONE",
                "raw_grip": "NONE",
                "confidence": 0.0,
                "probabilities": {},
                "alignment": {}
            }

        # Orientation gating: Only classify when back of hand is facing camera
        orientation = features.get("orientation", "")
        if orientation != "BACK":
            return {
                "grip": "NONE",
                "raw_grip": "NONE",
                "confidence": 0.0,
                "probabilities": {},
                "alignment": {}
            }

        # Check required ball distance features (must not be None)
        if features.get("ball_index_distance") is None:
            return {
                "grip": "NONE",
                "raw_grip": "NONE",
                "confidence": 0.0,
                "probabilities": {},
                "alignment": {}
            }

        try:
            # 1. Convert feature dictionary into single-row DataFrame
            df = pd.DataFrame([features])

            # 2. Apply One-Hot Encoding for categorical fields (hand_side, orientation)
            df_encoded = pd.get_dummies(df)

            # 3. Check alignment diagnostics
            alignment = self.verify_alignment(df_encoded)

            # 4. Match saved 24-feature columns schema and fill missing dummy columns with 0
            df_aligned = df_encoded.reindex(columns=self.feature_columns, fill_value=0)

            # 5. Cast all columns to float for model compatibility
            df_aligned = df_aligned.astype(float)

            # 6. Predict class probabilities
            probabilities = self.model.predict_proba(df_aligned)[0]
            max_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[max_idx])

            # 7. Decode raw class label
            if hasattr(self.label_encoder, "inverse_transform"):
                raw_grip = self.label_encoder.inverse_transform([max_idx])[0]
            else:
                raw_grip = str(self.label_encoder.classes_[max_idx])

            # 8. Build full probability distribution (sorted highest to lowest)
            prob_map = {
                str(cls_name): round(float(prob), 4)
                for cls_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
            sorted_probs = dict(sorted(prob_map.items(), key=lambda item: item[1], reverse=True))

            # 9. Apply confidence threshold (below 0.60 / 60% -> UNCERTAIN)
            if confidence < self.confidence_threshold:
                final_grip = "UNCERTAIN"
            else:
                final_grip = raw_grip

            return {
                "grip": final_grip,
                "raw_grip": raw_grip,
                "confidence": round(confidence, 4),
                "probabilities": sorted_probs,
                "alignment": alignment
            }

        except Exception:
            return {
                "grip": "NONE",
                "raw_grip": "NONE",
                "confidence": 0.0,
                "probabilities": {},
                "alignment": {}
            }


def evaluate_dataset_samples(csv_path="data/features.csv", models_dir="models"):
    """
    Temporary debug function to test model predictions directly on dataset samples
    for all 5 classes to confirm model capability outside webcam inference.
    """
    print("=" * 60)
    print("  [DEBUG] Testing Dataset Sample Predictions (Direct Model Bypass)")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"CSV not found at: {csv_path}")
        return

    classifier = GripClassifier(models_dir=models_dir, confidence_threshold=0.60)
    df = pd.read_csv(csv_path)

    # Clean label mapping
    mask = df["label"].isna() & df["wrist_rotation_angle"].notna()
    if mask.any():
        df.loc[mask, "label"] = df.loc[mask, "wrist_rotation_angle"]
        df.loc[mask, "wrist_rotation_angle"] = np.nan

    target_classes = ["seam_grip", "off_spin_grip", "inswing", "outswing", "knuckle_ball"]

    for target in target_classes:
        sub = df[df["label"] == target]
        if len(sub) == 0:
            print(f"\n[Class: {target}] No samples found in CSV!")
            continue

        sample_row = sub.iloc[0].to_dict()
        sample_row["orientation"] = "BACK"  # Force BACK to test classifier directly

        # Impute missing curls from angles if needed
        if pd.isna(sample_row.get("index_curl")):
            sample_row["index_curl"] = sample_row.get("index_angle", 100.0)
        if pd.isna(sample_row.get("middle_curl")):
            sample_row["middle_curl"] = sample_row.get("middle_angle", 100.0)
        if pd.isna(sample_row.get("ring_curl")):
            sample_row["ring_curl"] = sample_row.get("ring_angle", 40.0)

        # Impute missing values with medians
        for k, v in sample_row.items():
            if pd.isna(v):
                sample_row[k] = 0.0

        res = classifier.predict(sample_row)

        print(f"\nTarget: {target.upper()}")
        print(f"  Predicted (Thresholded): {res['grip'].upper()}")
        print(f"  Raw Top Class:           {res['raw_grip'].upper()} ({res['confidence']*100:.1f}%)")
        print("  Probabilities:")
        for cls_name, prob in res["probabilities"].items():
            print(f"    - {cls_name:15s}: {prob*100:5.1f}%")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_dataset_samples()
