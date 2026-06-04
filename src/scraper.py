from google_play_scraper import Sort, reviews
import pandas as pd
from packaging import version
import time
import os

# ==============================
# 1. MENCARI N VERSI TERBARU
# ==============================
def get_latest_versions(app_id, n_versions, limit_no_new=15):
    print(f"Mengidentifikasi {n_versions} versi terbaru aplikasi DANA...") 

    found_versions = set()
    continuation_token = None
    no_new_version_count = 0

    while True:
        result, continuation_token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=200,
            continuation_token=continuation_token
        )

        if not result:
            break

        new_found = False

        for r in result:
            v = r.get('reviewCreatedVersion')
            if v and v.count('.') >= 2:
                if v not in found_versions:
                    found_versions.add(v)
                    new_found = True

        if new_found:
            no_new_version_count = 0
        else:
            no_new_version_count += 1

        print(f"Total versi aplikasi terdeteksi: {len(found_versions)}")

        # Stop jika tidak ada versi baru dalam beberapa batch
        if no_new_version_count >= limit_no_new:
            print("Versi sudah cukup lengkap")
            break

        if not continuation_token:
            break

        time.sleep(0.5)

    latest_versions = sorted(found_versions, key=version.parse, reverse=True)[:n_versions]

    print(f"\n{n_versions} Versi target:")
    for v in latest_versions:
        print("-", v)

    return latest_versions

# ==============================
# 2. SCRAPING REVIEW BERDASARKAN VERSI
# ==============================
def scrape_reviews(app_id, target_versions, limit_miss=20):
    print("\nMulai scraping review...")

    all_reviews = []
    continuation_token = None

    version_active = {v: True for v in target_versions}
    version_miss_count = {v: 0 for v in target_versions}

    while True:
        result, continuation_token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=200,
            continuation_token=continuation_token
        )

        if not result:
            break

        found_in_batch = {v: False for v in target_versions}

        for r in result:
            v = r.get('reviewCreatedVersion')
            if v in target_versions:
                all_reviews.append(r)
                found_in_batch[v] = True

        # Update status tiap versi
        for v in target_versions:
            if found_in_batch[v]:
                version_miss_count[v] = 0
            else:
                version_miss_count[v] += 1

            if version_miss_count[v] >= limit_miss:
                version_active[v] = False

        print("Status versi:", version_active)

        # Stop jika semua versi sudah habis
        if all(not active for active in version_active.values()):
            print("Scraping pada versi target telah selesai dilakukan")
            break

        if not continuation_token:
            break

        time.sleep(0.5)

    return all_reviews

# ==============================
# 3. SUMMARY STATISTIK
# ==============================
def show_summary(all_reviews):
    df = pd.DataFrame(all_reviews)

    if df.empty:
        print("Tidak ada data untuk ditampilkan.")
        return

    print("\nStatistik:")

    stats = df.groupby('reviewCreatedVersion').agg(
        jumlah=('content', 'count'),
        awal=('at', 'min'),
        akhir=('at', 'max')
    ).sort_index(ascending=False)

    print(stats)

# ==============================
# 4. MENYIMPAN DATA
# ==============================
def save_reviews(
    all_reviews,
    output_path=os.path.join(
        "data",
        "raw_reviews_20.csv"
    )
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(all_reviews).drop_duplicates(subset=['reviewId'])

    if df.empty:
        print("Tidak ada data ditemukan.")
        return

    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\nTotal ulasan: {len(df)}")
    print(f"Data disimpan di: {output_path}")


# ==============================
# 5. MAIN (ENTRY POINT)
# ==============================
if __name__ == "__main__":
    app_id = "id.dana"

    # Ubah jumlah versi di sini
    N_VERSIONS = 20

    versions = get_latest_versions(app_id, n_versions=N_VERSIONS)
    reviews_data = scrape_reviews(app_id, versions)
    
    # Menampilkan summary
    show_summary(reviews_data)

    # Menyimpan ke CSV
    save_reviews(reviews_data)