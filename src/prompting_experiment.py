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
INPUT_PATH = os.path.join("data", "unlabeled", "train_manual.csv")
EXAMPLE_SOURCE = os.path.join("data", "labeled", "pseudolabel", "selected_prompt_examples.csv")

VALID_METHODS = ["zero_shot", "one_shot", "few_shot"]
LABELS = ["accuracy", "completeness", "satisfaction"]

# Check if prompting method is provided
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
    raise ValueError(f"Metode tidak valid: {PROMPTING_METHOD}")

OUTPUT_DIR = os.path.join("data", "labeled", "pseudolabel", f"manual_{PROMPTING_METHOD}")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, f"manual_{PROMPTING_METHOD}_merged.csv")

BATCH_SIZE = 50
EXPERIMENT_SIZE = None
SLEEP_TIME = 3
MAX_RETRY = 3
RETRY_SLEEP_TIME = 30
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
    - Keluhan teknis dapat menghasilkan accuracy = 1.
    - Keluhan teknis yang disertai nada negatif dapat menghasilkan satisfaction = 1.
    - Dana atau saldo hilang menghasilkan accuracy = 1.
    - Dana cicilan atau fitur yang belum muncul menghasilkan completeness = 1.
    - Fitur tersedia tetapi tidak dapat digunakan menghasilkan accuracy = 1.
    - CS atau layanan bantuan yang tidak membantu dapat dianggap completeness = 1 dan satisfaction = 1.
    - Jika suatu aspek tidak relevan, beri 0.
    - Jangan melewatkan ulasan.
    - Jumlah output HARUS sama dengan jumlah input.

    Format output:
    [{"accuracy": 0, "completeness": 0, "satisfaction": 0}]

    Output HARUS berupa JSON valid tanpa teks tambahan.
    """

# Format Examples for one-shot & few-shot prompting
def format_examples(examples):
    text = ""

    for idx, example in enumerate(examples, start=1):
        text += f"""
        Contoh {idx} untuk label {example["target_label"]}:
        Ulasan:
        {example["review"]}

        Output:
        {{"accuracy":{example["accuracy"]},"completeness":{example["completeness"]},"satisfaction":{example["satisfaction"]}}}
        """

    return text

# Load selected examples
def load_prompt_examples():
    df = pd.read_csv(EXAMPLE_SOURCE)

    required_columns = [
        "target_label",
        "cleaned_content",
        "accuracy",
        "completeness",
        "satisfaction"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan di {EXAMPLE_SOURCE}"
            )

    def select_examples(n_per_label):
        examples = []

        for label in LABELS:
            selected = (
                df.loc[
                    df["target_label"] == label
                ]
                .head(n_per_label)
            )

            if len(selected) != n_per_label:
                raise ValueError(
                    f"Jumlah contoh untuk label '{label}' harus {n_per_label}."
                )

            for _, row in selected.iterrows():
                examples.append({
                    "target_label": label,
                    "review": row["cleaned_content"],
                    "accuracy": int(row["accuracy"]),
                    "completeness": int(row["completeness"]),
                    "satisfaction": int(row["satisfaction"])
                })

        return examples

    one_shot_examples = select_examples(n_per_label=1)
    few_shot_examples = select_examples(n_per_label=2)

    return one_shot_examples, few_shot_examples

ONE_SHOT_EXAMPLES, FEW_SHOT_EXAMPLES = load_prompt_examples()


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
    examples = format_examples(ONE_SHOT_EXAMPLES)

    return f"""
    {annotation_guideline()}

    Contoh anotasi:
    {examples}

    Data ulasan:
    {review_text}

    JSON:
    """

# Few-shot prompting
def few_shot_prompt(review_text):
    examples = format_examples(FEW_SHOT_EXAMPLES)

    return f"""
    {annotation_guideline()}

    Contoh anotasi:
    {examples}

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
            
            if not isinstance(outputs, list):
                print("\nOutput bukan list JSON")
                continue

            # validate output length
            if len(outputs) != len(reviews):
                print("\nJumlah output tidak sesuai!")
                print(f"Expected : {len(reviews)}")
                print(f"Received : {len(outputs)}")
                continue

            return outputs

        except Exception as e:
            print(f"\nError batch attempt {attempt+1}")
            print(e)
            print(f"Menunggu {RETRY_SLEEP_TIME} detik sebelum mencoba kembali...")
            time.sleep(RETRY_SLEEP_TIME)

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
    for label in LABELS:
        print(
            f"{label}: "
            f"{final_df[label].sum()}"
        )

# Main labeling
def run_experiment():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    if EXPERIMENT_SIZE is not None:
        df = df.head(EXPERIMENT_SIZE)

    total_data = len(df)
    total_batches = math.ceil(total_data / BATCH_SIZE)

    print(f"Prompting method : {PROMPTING_METHOD}")
    print(f"Total data   : {total_data}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Total batch  : {total_batches}")

    if PROMPTING_METHOD == "one_shot":
        print("\nContoh one-shot yang digunakan:")
        print(format_examples(ONE_SHOT_EXAMPLES))

    elif PROMPTING_METHOD == "few_shot":
        print("\nContoh few-shot yang digunakan:")
        print(format_examples(FEW_SHOT_EXAMPLES))

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

        # skip batch that already exist
        if os.path.exists(batch_file):
            print(f"Skip existing batch {batch_idx}")
            continue

        success = process_batch(batch_df, batch_idx)

        if not success:
            print(f"Skip batch {batch_idx} karena error")
            continue

        # rate limit safety
        time.sleep(SLEEP_TIME)


    # Check completion
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

# MAIN
if __name__ == "__main__":
    run_experiment()