import pandas as pd
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    accuracy_score
)

# Usability aspects used in the multi-label classification
LABELS = [
    "accuracy",
    "completeness",
    "satisfaction"
]

# Ground truth data path
GROUND_TRUTH = os.path.join(
    "data",
    "labeled",
    "manual",
    "train_manual_ground_truth.csv"
)

# Pseudolabel data paths for each prompting method
METHOD_FILES = {
    "zero_shot": os.path.join(
        "data",
        "labeled",
        "pseudolabel",
        "manual_zero_shot",
        "manual_zero_shot_merged.csv"
    ),

    "one_shot": os.path.join(
        "data",
        "labeled",
        "pseudolabel",
        "manual_one_shot",
        "manual_one_shot_merged.csv"
    ),

    "few_shot": os.path.join(
        "data",
        "labeled",
        "pseudolabel",
        "manual_few_shot",
        "manual_few_shot_merged.csv"
    )
}

# Output file path
OUTPUT_FILE = os.path.join(
    "data",
    "labeled",
    "pseudolabel",
    "prompting_comparison.csv"
)

# Validate if file exists
def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {file_path}"
        )

# Evaluate each prompting method by comparing it to the ground truth
def evaluate_method(method_name, prediction_file, ground_truth_file):
    print("\n" + "=" * 60)
    print(f"Prompting method: {method_name}")
    print("=" * 60)

    validate_file_exists(ground_truth_file)
    validate_file_exists(prediction_file)

    gt = pd.read_csv(ground_truth_file)
    pred = pd.read_csv(prediction_file)

    # Take only the first n rows where n is the number of rows in the prediction file
    gt = gt.head(len(pred))

    # Validate if the number of rows is the same
    if len(gt) != len(pred):
        raise ValueError(f"Jumlah data tidak sama ({len(gt)} vs {len(pred)})")

    print(f"Jumlah data yang dibandingkan: {len(pred)}")

    # Ensure both files contain the same reviews in the same order
    if not (
        gt["cleaned_content"]
        .reset_index(drop=True)
        .equals(
            pred["cleaned_content"]
            .reset_index(drop=True)
        )
    ):
        raise ValueError(
            "Urutan atau isi ulasan tidak sama."
        )

    print("Validasi ulasan berhasil")
    
    y_true = gt[LABELS]
    y_pred = pred[LABELS]

    results = []

    # Calculate metrics for each label
    for label in LABELS:
        precision = precision_score(
            y_true[label],
            y_pred[label],
            zero_division=0
        )

        recall = recall_score(
            y_true[label],
            y_pred[label],
            zero_division=0
        )

        f1 = f1_score(
            y_true[label],
            y_pred[label],
            zero_division=0
        )

        results.append({
            "label": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

    result_df = pd.DataFrame(results)

    print("\nPer Label:")
    print(result_df)

    label_f1 = {
        f"{label}_f1": result_df.loc[
            result_df["label"] == label,
            "f1_score"
        ].values[0]
        for label in LABELS
    }
    
    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    exact_match_ratio = accuracy_score(
        y_true,
        y_pred
    )

    hamming = hamming_loss(
        y_true,
        y_pred
    )

    print("\nOverall:")
    print(f"Micro Precision : {micro_precision:.4f}")
    print(f"Micro Recall    : {micro_recall:.4f}")
    print(f"Micro F1        : {micro_f1:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Exact Match Ratio : {exact_match_ratio:.4f}")
    print(f"Hamming Loss      : {hamming:.4f}")

    return {
        "method": method_name,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match_ratio": round(exact_match_ratio, 4),
        "hamming_loss": round(hamming, 4),
        **label_f1
    }

def main():
    summary = []

    for method_name, file_path in METHOD_FILES.items():
        result = evaluate_method(
            method_name,
            file_path,
            GROUND_TRUTH
        )
        summary.append(result)

    summary_df = pd.DataFrame(summary)

    # Rank methods by Macro F1, Micro F1, and Hamming Loss
    summary_df = summary_df.sort_values(
        by=[
            "macro_f1",
            "micro_f1",
            "hamming_loss"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    print("\n")
    print("=" * 60)
    print("FINAL RANKING")
    print("=" * 60)

    print(summary_df)

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nHasil disimpan ke:")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    main()