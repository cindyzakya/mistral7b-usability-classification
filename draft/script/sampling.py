import pandas as pd
import os

# ==============================
# HITUNG MINIMUM (KELIPATAN 50)
# ==============================
def get_rounded_min(dist_series, base=50):
    raw_min = dist_series.min()

    # Bulatkan ke bawah kelipatan 50
    rounded_min = (raw_min // base) * base

    # Jaga agar tidak 0
    if rounded_min == 0:
        rounded_min = base

    return raw_min, rounded_min


# ==============================
# STRATIFIED SAMPLING
# ==============================
def stratified_sampling(input_path, output_path):
    print("Loading cleaned data...")

    df = pd.read_csv(input_path)

    print("Distribusi awal:")
    dist_before = df['score'].value_counts().sort_index()
    print(dist_before)

    # ==============================
    # HITUNG MINIMUM (BULAT)
    # ==============================
    raw_min, min_count = get_rounded_min(dist_before)

    print(f"\nJumlah minimum asli: {raw_min}")
    print(f"Jumlah minimum dibulatkan ke bawah (kelipatan 50): {min_count}")

    # ==============================
    # STRATIFIED SAMPLING
    # ==============================
    print("\nMelakukan stratified sampling...")

    df_sampled = (
        df.groupby('score', group_keys=False)
        .sample(n=min_count, random_state=42)
    )

    df_sampled = df_sampled.reset_index(drop=True)

    # ==============================
    # DISTRIBUSI SETELAH SAMPLING
    # ==============================
    print("Distribusi setelah sampling:")
    dist_after = df_sampled['score'].value_counts().sort_index()
    print(dist_after)

    print(f"\nTotal data setelah sampling: {len(df_sampled)}")

    # ==============================
    # SIMPAN DATA
    # ==============================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sampled.to_csv(output_path, index=False, encoding='utf-8')

    print(f"Data disimpan di: {output_path}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    stratified_sampling(
        input_path="data/cleaned_reviews.csv",
        output_path="data/sampled_reviews.csv"
    )