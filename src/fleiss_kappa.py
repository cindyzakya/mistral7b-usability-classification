import os
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

# Usability aspects evaluated for inter-annotator agreement
LABELS = [
    "accuracy",
    "completeness",
    "satisfaction"
]


def interpret_kappa(kappa):
    if kappa < 0:
        return "Poor"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"

# Build the Fleiss' Kappa table
def build_fleiss_table(df1, df2, df3, label):
    table = []

    for i in range(len(df1)):
        votes = [
            int(df1.iloc[i][label]),
            int(df2.iloc[i][label]),
            int(df3.iloc[i][label])
        ]

        count_0 = votes.count(0)
        count_1 = votes.count(1)

        table.append([count_0, count_1])

    return np.array(table)

def load_annotator_data(folder_path):

    csv_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".csv")
        and "fleiss_kappa_result" not in f
    ])

    print("\nFiles detected:")

    for f in csv_files:
        print("-", f)

    # Handle pilot annotation files
    if len(csv_files) == 3:

        print("\nPilot Study")
        df1 = pd.read_csv(os.path.join(folder_path, csv_files[0]))
        df2 = pd.read_csv(os.path.join(folder_path, csv_files[1]))
        df3 = pd.read_csv(os.path.join(folder_path, csv_files[2]))

        return df1, df2, df3

    # Handle the complete annotation dataset
    elif len(csv_files) == 9:

        print("\nFinal Study")

        a1_files = sorted([
            f for f in csv_files
            if "_a1" in f
        ])

        a2_files = sorted([
            f for f in csv_files
            if "_a2" in f
        ])

        a3_files = sorted([
            f for f in csv_files
            if "_a3" in f
        ])

        # Ensure each annotator has exactly 3 same files
        if not (
            len(a1_files) == 3
            and len(a2_files) == 3
            and len(a3_files) == 3
        ):
            raise ValueError(
                "Setiap annotator harus memiliki tepat 3 file "
                "(test, validation, train_manual)."
            )

        print("\nAnnotator A1:")
        for f in a1_files:
            print("-", f)

        print("\nAnnotator A2:")
        for f in a2_files:
            print("-", f)

        print("\nAnnotator A3:")
        for f in a3_files:
            print("-", f)

        df1 = pd.concat(
            [
                pd.read_csv(os.path.join(folder_path, f))
                for f in a1_files
            ],
            ignore_index=True
        )

        df2 = pd.concat(
            [
                pd.read_csv(os.path.join(folder_path, f))
                for f in a2_files
            ],
            ignore_index=True
        )

        df3 = pd.concat(
            [
                pd.read_csv(os.path.join(folder_path, f))
                for f in a3_files
            ],
            ignore_index=True
        )

        return df1, df2, df3

    else:

        raise ValueError(
            f"""
            Jumlah file tidak sesuai.

            Pilot Study:
            - 3 file anotator

            Final Study:
            - 9 file anotator

            Jumlah file ditemukan:
            {len(csv_files)}
            """
        )

def calculate_kappa(folder_path):

    df1, df2, df3 = load_annotator_data(
        folder_path
    )

    # Cheeck number of rows
    if not (
        len(df1) == len(df2) == len(df3)
    ):
        raise ValueError(
            "Jumlah baris ketiga annotator tidak sama."
        )

    # Check order of review
    if not (
        df1["cleaned_content"].equals(df2["cleaned_content"])
        and
        df1["cleaned_content"].equals(df3["cleaned_content"])
    ):
        raise ValueError(
            "Urutan atau isi review antar anotator tidak sama."
        )

    print(f"\nTotal data: {len(df1)}")

    results = []

    print("\n=== FLEISS' KAPPA ===")
    for label in LABELS:
        table = build_fleiss_table(
            df1,
            df2,
            df3,
            label
        )

        kappa = fleiss_kappa(table)

        interpretation = interpret_kappa(
            kappa
        )

        results.append({
            "label": label,
            "fleiss_kappa": round(kappa, 4),
            "interpretation": interpretation
        })

        print(
            f"{label:<15} : "
            f"{kappa:.4f} "
            f"({interpretation})"
        )

    result_df = pd.DataFrame(results)

    folder_name = os.path.basename(
        folder_path
    )

    output_file = os.path.join(
        folder_path,
        f"{folder_name}_fleiss_kappa_result.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nHasil disimpan ke:\n{output_file}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Folder data belum diberikan.\n"
            "Silakan jalankan program dengan format berikut:\n"
            "python src/fleiss_kappa.py <path_folder>\n\n"
            "Contoh:\n"
            "python src/fleiss_kappa.py data/labeled/manual/pilot"
        )

        sys.exit()

    folder_path = sys.argv[1]

    calculate_kappa(folder_path)