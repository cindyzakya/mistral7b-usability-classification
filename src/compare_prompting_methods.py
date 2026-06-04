import pandas as pd
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

LABELS = [
    "accuracy",
    "completeness",
    "satisfaction"
]

GROUND_TRUTH = os.path.join(
    "data",
    "labeled",
    "manual",
    "train_manual_ground_truth.csv"
)

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

OUTPUT_FILE = os.path.join(
    "data",
    "labeled",
    "pseudolabel",
    "prompting_comparison.csv"
)

def evaluate_method(
    method_name,
    prediction_file,
    ground_truth_file
):

    print("\n" + "=" * 60)
    print(f"Prompting method: {method_name}")
    print("=" * 60)

    gt = pd.read_csv(
        ground_truth_file
    )

    pred = pd.read_csv(
        prediction_file
    )

    gt = gt.head(len(pred))

    if len(gt) != len(pred):
        raise ValueError(
            f"Jumlah data tidak sama "
            f"({len(gt)} vs {len(pred)})"
        )
    else:
        print(
            f"Jumlah data yang dibandingkan: "
            f"{len(pred)}"
        )

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

    print(
        "Validasi ulasan berhasil."
    )

    y_true = gt[LABELS]
    y_pred = pred[LABELS]

    results = []

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
            "precision": round(
                precision,
                4
            ),
            "recall": round(
                recall,
                4
            ),
            "f1_score": round(
                f1,
                4
            )
        })

    result_df = pd.DataFrame(
        results
    )

    print("\nPer Label:")
    print(result_df)

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

    print("\nOverall:")

    print(
        f"Micro Precision : "
        f"{micro_precision:.4f}"
    )

    print(
        f"Micro Recall    : "
        f"{micro_recall:.4f}"
    )

    print(
        f"Micro F1        : "
        f"{micro_f1:.4f}"
    )

    return {
        "method": method_name,
        "micro_precision":
            micro_precision,
        "micro_recall":
            micro_recall,
        "micro_f1":
            micro_f1
    }

def main():
    summary = []
    for method_name, file_path in (METHOD_FILES.items()):
        result = evaluate_method(
            method_name,
            file_path,
            GROUND_TRUTH
        )
        summary.append(result)

    summary_df = pd.DataFrame(summary)

    summary_df = (
        summary_df
        .sort_values(
            by="micro_f1",
            ascending=False
        )
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