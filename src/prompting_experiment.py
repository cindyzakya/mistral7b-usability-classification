import pandas as pd
import os
import sys
import time
import json
import glob
import math

from dotenv import load_dotenv
from google import genai

# Config
INPUT_PATH = os.path.join(
    "data",
    "unlabeled",
    "train_manual.csv"
)

VALID_METHODS = [
    "zero_shot",
    "one_shot",
    "few_shot"
]

if len(sys.argv) != 2:
    print(
        "Metode prompting belum diberikan.\n\n"
        "Contoh:\n"
        "python src/prompting_experiment.py zero_shot\n"
        "python src/prompting_experiment.py one_shot\n"
        "python src/prompting_experiment.py few_shot"
    )
    sys.exit()

PROMPTING_METHOD = sys.argv[1]
if PROMPTING_METHOD not in VALID_METHODS:
    raise ValueError(
        f"Metode tidak valid: {PROMPTING_METHOD}"
    )

OUTPUT_DIR = os.path.join(
    "data",
    "labeled",
    "pseudolabel",
    f"manual_{PROMPTING_METHOD}"
)

FINAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    f"manual_{PROMPTING_METHOD}_merged.csv"
)

BATCH_SIZE = 50
EXPERIMENT_SIZE = 500
SLEEP_TIME = 3
MAX_RETRY = 3

MODEL_NAME = "gemini-3.5-flash"

# Load env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

# Gemini Client
client = genai.Client(api_key=API_KEY)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Annotation guideline
def annotation_guideline():
    return """
    Anda adalah anotator usability.
    Tugas:
    Klasifikasikan setiap ulasan pengguna menggunakan multi-label classification ke dalam tiga label berikut:

    1. accuracy
    Label = 1 jika terdapat masalah yang menunjukkan kegagalan sistem, proses, transaksi, atau fungsi aplikasi dalam membantu pengguna mencapai tujuan penggunaan.
    Contoh:
    - error
    - bug
    - crash
    - force close
    - login gagal
    - transaksi gagal
    - OTP gagal
    - QRIS gagal
    - saldo atau dana hilang
    - pending
    - loading berkepanjangan
    - delay
    - verifikasi gagal
    - fitur tersedia tetapi tidak dapat digunakan
    - sistem menghasilkan hasil yang tidak sesuai harapan pengguna
    Fokus accuracy adalah fungsi aplikasi yang tidak berjalan sebagaimana mestinya.

    --------------------------------------------------

    2. completeness
    Label = 1 jika pengguna mengeluhkan bahwa kebutuhan penggunaan belum didukung secara memadai oleh aplikasi.
    Contoh:
    - fitur tidak tersedia
    - fitur belum muncul
    - fitur kurang lengkap
    - membutuhkan fitur tambahan
    - layanan bantuan atau CS tidak membantu
    - proses upgrade akun dipersulit
    - opsi login atau akses dianggap tidak memadai
    - layanan atau fungsi yang dibutuhkan pengguna belum tersedia
    - proses penggunaan dianggap tidak lengkap atau tidak mendukung kebutuhan pengguna
    Fokus completeness adalah kelengkapan dukungan aplikasi terhadap kebutuhan pengguna.

    --------------------------------------------------

    3. satisfaction
    Label = 1 jika terdapat ketidakpuasan, frustrasi, kekecewaan, kemarahan, keluhan bernada negatif, atau evaluasi negatif terhadap pengalaman penggunaan aplikasi.
    Contoh:
    - kecewa
    - kesal
    - marah
    - ribet
    - menyulitkan
    - membingungkan
    - buruk
    - parah
    - sangat mengecewakan
    Label satisfaction dapat diberikan meskipun tidak terdapat kata emosi secara eksplisit apabila isi ulasan secara keseluruhan menunjukkan pengalaman penggunaan yang negatif.

    --------------------------------------------------

    Aturan penting:
    - Satu ulasan dapat memiliki lebih dari satu label.
    - Ulasan yang tidak berkaitan dengan usability dapat diberi semua label 0.
    - Label diberikan berdasarkan isi utama ulasan.
    - Fokus pada makna keseluruhan ulasan, bukan hanya keyword tertentu.
    - Keluhan teknis sering menghasilkan accuracy = 1.
    - Keluhan teknis yang disertai nada negatif dapat menghasilkan satisfaction = 1.
    - Dana atau saldo hilang biasanya accuracy = 1.
    - Dana cicilan atau fitur yang belum muncul biasanya completeness = 1.
    - Fitur tersedia tetapi tidak dapat digunakan biasanya accuracy = 1.
    - CS atau layanan bantuan yang tidak membantu dapat dianggap completeness = 1 dan satisfaction = 1.
    - Jika suatu aspek tidak relevan, beri 0.
    - Jangan melewatkan ulasan.
    - Jumlah output HARUS sama dengan jumlah input.

    Format output:
    [
    {
        "accuracy": 0,
        "completeness": 0,
        "satisfaction": 0
    }
    ]

    Output HARUS berupa JSON valid tanpa teks tambahan.
    """

# Zero-shot prompting
def zero_shot_prompt(review_text):

    return f"""
    {annotation_guideline()}

    Data ulasan:
    {review_text}

    JSON:
    """

# One-shot prompting
def one_shot_prompt(review_text):

    return f"""
    {annotation_guideline()}

    Contoh:
    Ulasan:
    "Aplikasi sering error saat transfer"

    Output:
    {{
        "accuracy": 1,
        "completeness": 0,
        "satisfaction": 0
    }}

    Data ulasan:
    {review_text}

    JSON:
    """

# Few-shot prompting
def few_shot_prompt(review_text):

    return f"""
    {annotation_guideline()}

    Contoh:

    Ulasan 0:
    "Aplikasi sering error saat transfer"
    Output:
    {{"accuracy":1,"completeness":0,"satisfaction":0}}

    Ulasan 1:
    "Fitur cicilan belum muncul"
    Output:
    {{"accuracy":0,"completeness":1,"satisfaction":0}}

    Ulasan 2:
    "CS tidak membantu sama sekali"
    Output:
    {{"accuracy":0,"completeness":1,"satisfaction":1}}

    Ulasan 3:
    "Dana saya hilang dan sangat mengecewakan"
    Output:
    {{"accuracy":1,"completeness":0,"satisfaction":1}}

    Ulasan 4:
    "Aplikasi parah, loading terus"
    Output:
    {{"accuracy":1,"completeness":0,"satisfaction":1}}

    Ulasan 5:
    "OTP wajib padahal nomor lama sudah hilang"
    Output:
    {{"accuracy":0,"completeness":1,"satisfaction":0}}

    Data ulasan:
    {review_text}

    JSON:
    """

# Create prompt
def create_batch_prompt(reviews):

    review_text = ""

    for i, review in enumerate(reviews):
        review_text += f"""
    Review {i}:
    \"\"\"{review}\"\"\"
    """

    if PROMPTING_METHOD == "zero_shot":
        return zero_shot_prompt(review_text)
    elif PROMPTING_METHOD == "one_shot":
        return one_shot_prompt(review_text)
    elif PROMPTING_METHOD == "few_shot":
        return few_shot_prompt(review_text)
    else:
        raise ValueError("PROMPTING_METHOD tidak valid")

# Gemini call
def label_batch(reviews):

    prompt = create_batch_prompt(reviews)

    for attempt in range(MAX_RETRY):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0,
                    "top_p": 0.1,
                    "response_mime_type": "application/json"
                }
            )

            outputs = json.loads(response.text)
            
            # validasi jumlah output
            if len(outputs) != len(reviews):
                print("\nJumlah output tidak sesuai!")
                print(f"Expected : {len(reviews)}")
                print(f"Received : {len(outputs)}")

                return None

            return outputs

        except Exception as e:
            print(f"\nError batch attempt {attempt+1}")
            print(e)

            time.sleep(5)

    return None

# Process batch
def process_batch(batch_df, batch_idx):

    reviews = batch_df["cleaned_content"].fillna("").tolist()

    print(f"\nProcessing batch {batch_idx}")

    outputs = label_batch(reviews)

    if outputs is None:
        print(f"Batch {batch_idx} gagal")

        return False

    results = []

    for (_, row), output in zip(batch_df.iterrows(), outputs):

        results.append({
            "content": row.get("content", ""),
            "cleaned_content": row.get("cleaned_content", ""),
            "score": row.get("score", ""),
            "accuracy": int(output.get("accuracy", 0)),
            "completeness": int(output.get("completeness", 0)),
            "satisfaction": int(output.get("satisfaction", 0)),
        })

    # save per batch
    batch_file = os.path.join(
        OUTPUT_DIR,
        f"batch_{batch_idx:04d}.csv"
    )

    pd.DataFrame(results).to_csv(
        batch_file,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved : {batch_file}")

    return True

# Merge all batches
def merge_batches():

    print("\nMerging all batches...")

    batch_files = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "batch_*.csv"))
    )

    if not batch_files:

        print("Tidak ada batch file")

        return

    all_df = []

    for file in batch_files:

        df = pd.read_csv(file)

        all_df.append(df)

    final_df = pd.concat(all_df, ignore_index=True)

    final_df.to_csv(
        FINAL_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    print(f"\nFinal file saved : {FINAL_OUTPUT}")
    print(f"Total labeled data : {len(final_df)}")
    print("\nDistribusi label:")
    for label in [
        "accuracy",
        "completeness",
        "satisfaction"
    ]:

        print(
            f"{label}: "
            f"{final_df[label].sum()}"
        )

# =========================================================
# MAIN LABELING
# =========================================================
def run_experiment():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)
    df = df.head(EXPERIMENT_SIZE)

    total_data = len(df)

    total_batches = math.ceil(total_data / BATCH_SIZE)

    print(f"Prompting method : {PROMPTING_METHOD}")
    print(f"Total data   : {total_data}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Total batch  : {total_batches}")

    for batch_idx in range(total_batches):

        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE

        batch_df = df.iloc[start:end]

        if len(batch_df) == 0:
            continue

        batch_file = os.path.join(
            OUTPUT_DIR,
            f"batch_{batch_idx:04d}.csv"
        )

        # skip jika batch sudah ada
        if os.path.exists(batch_file):

            print(f"Skip existing batch {batch_idx}")

            continue

        success = process_batch(batch_df, batch_idx)

        if not success:

            print(f"Skip batch {batch_idx} karena error")

            continue

        # rate limit safety
        time.sleep(SLEEP_TIME)

    # =====================================================
    # CHECK COMPLETION
    # =====================================================
    existing_batches = len(
        glob.glob(os.path.join(OUTPUT_DIR, "batch_*.csv"))
    )

    print(f"\nFinished batches : {existing_batches}/{total_batches}")

    if existing_batches == total_batches:

        print("\nSemua batch selesai")
        merge_batches()

    else:

        print("\nMasih ada batch yang belum selesai")
        print("Silakan jalankan ulang script untuk melanjutkan proses labeling")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    run_experiment()