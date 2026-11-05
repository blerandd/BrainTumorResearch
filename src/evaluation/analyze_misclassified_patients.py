import pandas as pd

path = "results/classification/patient_level/image_predictions_with_patient_ids.csv"

df = pd.read_csv(path)
df["patient_id"] = df["patient_id"].astype(str).str.strip()

wrong_patients = [
    "100820",
    "101020",
    "102929",
    "110116",
    "113971",
    "92897",
    "98889"
]

columns = [
    "sample_id",
    "patient_id",
    "true_label",
    "predicted_label",
    "glioma_probability",
    "meningioma_probability",
    "pituitary_probability",
    "correct"
]

errors = df[df["patient_id"].isin(wrong_patients)].copy()
errors = errors.sort_values(["patient_id", "sample_id"])

print("\nINDIVIDUAL MRI PREDICTIONS")
print("=" * 100)
print(errors[columns].to_string(index=False))

print("\nPATIENT-LEVEL PROBABILITY ANALYSIS")
print("=" * 100)

summary = errors.groupby("patient_id").agg(
    true_label=("true_label", "first"),
    total_images=("sample_id", "count"),
    correctly_classified=("correct", "sum"),
    average_glioma=("glioma_probability", "mean"),
    average_meningioma=("meningioma_probability", "mean"),
    average_pituitary=("pituitary_probability", "mean")
)

print(summary.round(4).to_string())

output = "results/classification/patient_level/misclassified_patient_analysis.csv"
errors[columns].to_csv(output, index=False)

print(f"\nSaved detailed analysis to: {output}")