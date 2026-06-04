import os
import pandas as pd

LABELS = [
    "accuracy",
    "completeness",
    "satisfaction"
]

def majority_vote(a, b, c):
    total_vote = a + b + c

    if total_vote >= 2:
        return 1

    return 0

def create_ground_truth(
    file_a1,
    file_a2,
    file_a3,
    output_file
):

    df1 = pd.read_csv(file_a1)
    df2 = pd.read_csv(file_a2)
    df3 = pd.read_csv(file_a3)

    if not (
        len(df1) == len(df2) == len(df3)
    ):
        raise ValueError(
            "Jumlah baris ketiga annotator tidak sama."
        )

    if not (
        df1["cleaned_content"].equals(df2["cleaned_content"])
        and
        df1["cleaned_content"].equals(df3["cleaned_content"])
    ):
        raise ValueError(
            "Urutan review antar annotator tidak sama."
        )

    for label in LABELS:
        if (
            df1[label].isna().any()
            or df2[label].isna().any()
            or df3[label].isna().any()
        ):
            raise ValueError(
                f"Masih terdapat nilai kosong pada kolom '{label}'."
            )

    result = df1.copy()
    print(f"\nMemproses {len(result)} data...")

    for label in LABELS:

        result[label] = [

            majority_vote(a, b, c)

            for a, b, c in zip(
                df1[label],
                df2[label],
                df3[label]
            )
        ]

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Ground truth berhasil disimpan:\n{output_file}"
    )


if __name__ == "__main__":

    BASE = os.path.join(
        "data",
        "labeled",
        "manual",
        "final"
    )

    OUTPUT = os.path.join(
        "data",
        "labeled",
        "manual"
    )

    create_ground_truth(
        os.path.join(BASE, "test_a1.csv"),
        os.path.join(BASE, "test_a2.csv"),
        os.path.join(BASE, "test_a3.csv"),
        os.path.join(OUTPUT, "test_ground_truth.csv")
    )

    create_ground_truth(
        os.path.join(BASE, "validation_a1.csv"),
        os.path.join(BASE, "validation_a2.csv"),
        os.path.join(BASE, "validation_a3.csv"),
        os.path.join(OUTPUT, "validation_ground_truth.csv")
    )

    create_ground_truth(
        os.path.join(BASE, "train_manual_a1.csv"),
        os.path.join(BASE, "train_manual_a2.csv"),
        os.path.join(BASE, "train_manual_a3.csv"),
        os.path.join(OUTPUT, "train_manual_ground_truth.csv")
    )

    print("\nSemua ground truth berhasil dibuat.")