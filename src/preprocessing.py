import pandas as pd
import re
import os
import numpy as np

# ==============================
# CLEANING FUNCTION
# ==============================
def clean_text(text):
    if pd.isna(text):
        return ""

    # 1. Case folding
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Hapus karakter aneh (tanda baca dasar dipertahankan)
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)

    # 4. Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================
# FILTERING (RELEVANSI)
# ==============================
def is_relevant(text, min_length=4):
    words = text.split()

    # terlalu pendek
    if len(words) < min_length:
        return False

    # tidak ada huruf
    if not re.search(r"[a-z]", text):
        return False

    return True


# ==============================
# VALIDASI VERSION
# ==============================
def valid_version(v):
    if pd.isna(v):
        return False
    return isinstance(v, str) and v.count('.') >= 2


# ==============================
# FILTER VERSION BERDASARKAN JUMLAH REVIEW
# ==============================
def filter_versions_by_review_count(df):

    print("\nMenghitung jumlah review per versi...")

    # hitung jumlah review tiap versi
    counts = df['reviewCreatedVersion'].value_counts()

    print("\nJumlah review per versi:")
    print(counts)

    # ==============================
    # LOG TRANSFORMATION
    # ==============================
    log_counts = np.log10(counts.values)

    # ==============================
    # IQR PADA DATA LOG
    # ==============================
    Q1 = np.percentile(log_counts, 25)
    Q3 = np.percentile(log_counts, 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    # kembali ke skala asli
    threshold = 10 ** lower_bound

    print("\n===== HASIL DETEKSI OUTLIER BERBASIS LOG-IQR =====")
    print(f"Q1 (log)           : {Q1:.3f}")
    print(f"Q3 (log)           : {Q3:.3f}")
    print(f"IQR                : {IQR:.3f}")
    print(f"Lower Bound (log)  : {lower_bound:.3f}")
    print(f"Threshold review   : {threshold:.2f}")

    # ==============================
    # DETEKSI OUTLIER RENDAH
    # ==============================
    outlier_versions = counts[counts < threshold]

    print("\nVersi yang dianggap outlier rendah:")
    for v in outlier_versions.index:
        print(f"- {v} ({counts[v]} review)")

    # ==============================
    # AMBIL VERSI VALID
    # ==============================
    valid_versions = counts[counts >= threshold].index.tolist()

    print("\nVersi yang digunakan:")
    for v in valid_versions:
        print(f"- {v} ({counts[v]} review)")

    # ==============================
    # FILTER DATAFRAME
    # ==============================
    df_filtered = df[
        df['reviewCreatedVersion'].isin(valid_versions)
    ].reset_index(drop=True)

    print(f"\nTotal data setelah seleksi versi: {len(df_filtered)}")

    return df_filtered


# ==============================
# MAIN PREPROCESSING
# ==============================
def preprocess(input_path, output_path):
    print("Loading data...")

    df = pd.read_csv(input_path)

    print(f"Total data awal: {len(df)}")

    # Ambil kolom penting
    df = df[['content', 'reviewCreatedVersion', 'score']]

    # Drop NA
    df = df.dropna(subset=['content'])

    # ==============================
    # FILTER RATING (<= 3)
    # ==============================
    df = df[df['score'] <= 3]
    print(f"Total setelah filter rating <= 3: {len(df)}")

    # Validasi versi
    df = df[df['reviewCreatedVersion'].apply(valid_version)]

    # Cleaning
    print("Cleaning text...")
    df['cleaned_content'] = df['content'].apply(clean_text)

    # Hapus duplikasi
    df = df.drop_duplicates(subset=['cleaned_content'])

    # Filtering relevansi
    print("Memfilter review tidak relevan...")
    df = df[df['cleaned_content'].apply(is_relevant)]

    print(f"Total setelah preprocessing awal: {len(df)}")

    df = filter_versions_by_review_count(df)

    # Simpan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"Data disimpan di: {output_path}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    preprocess(
        input_path=os.path.join(
            "data",
            "raw_reviews_20.csv"
        ),
        output_path=os.path.join(
            "data",
            "cleaned_reviews_20.csv"
        )
    )