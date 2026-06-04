import pandas as pd
import os
import time
import json
from dotenv import load_dotenv
from groq import Groq

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# PROMPT
# ==============================
def create_prompt(review):
    return f"""
Klasifikasikan ulasan berikut berdasarkan aspek usability menurut ISO 9241-11.

Definisi:
- accuracy = 1 jika pengguna mengalami kesalahan fungsi (error, bug, gagal, tidak berjalan)
- completeness = 1 jika pengguna menyebutkan fitur tidak lengkap, kurang, atau kebutuhan fitur
- satisfaction = 1 jika terdapat respon emosional pengguna (puas, kecewa, marah, dll)

Aturan:
- Jika aspek tidak disebutkan → beri 0
- Satu ulasan bisa memiliki lebih dari satu label (multi-label)

Output HARUS berupa JSON VALID tanpa teks tambahan:

{{
  "accuracy": 0 atau 1,
  "completeness": 0 atau 1,
  "satisfaction": 0 atau 1,
  "reasoning": "penjelasan singkat berdasarkan isi ulasan (alasan pelabelan)"
}}

Ulasan:
"{review}"

JSON:
"""

# ==============================
# CALL GROQ + RETRY
# ==============================
def label_review(review, max_retry=3):
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": create_prompt(review)}
                ],
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error (attempt {attempt+1}):", e)
            time.sleep(2)
    
    return None

# ==============================
# PARSE JSON
# ==============================
def parse_output(output):
    if output is None:
        return None, None, None, None

    try:
        # Mengambil hanya bagian JSON
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]

        data = json.loads(json_str)

        return (
            int(data.get("accuracy", 0)),
            int(data.get("completeness", 0)),
            int(data.get("satisfaction", 0)),
            data.get("reasoning", "")
        )

    except Exception as e:
        print("Parse error:", e)
        return None, None, None, None

# ==============================
# MAIN LABELING
# ==============================
def labeling(input_path, output_path, limit=None, save_interval=10):
    print("Loading sampled data...")

    df = pd.read_csv(input_path)

    if limit:
        df = df.head(limit)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # resume support
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        results = existing_df.to_dict(orient="records")
        start_index = len(results)
        df = df.iloc[start_index:]
        print(f"Melanjutkan labeling dari data ke-{start_index}")
    else:
        results = []
        start_index = 0

    print(f"Total data yang diproses: {len(df)}")

    for i, row in df.iterrows():
        review = row['cleaned_content']

        print(f"[{start_index + i + 1}/{len(df)}] Processing...")

        output = label_review(review)

        acc, comp, sat, reason = parse_output(output)

        results.append({
            "content": row['content'],
            "cleaned_content": review,
            "score": row['score'],
            "accuracy": acc,
            "completeness": comp,
            "satisfaction": sat,
            "reasoning": reason
        })

        # save berkala
        if (i + 1) % save_interval == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_path, index=False)

        time.sleep(1)  # rate limit safety

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\nData berhasil disimpan di: {output_path}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    labeling(
        input_path="data/sampled_reviews.csv",
        output_path="data/labeled_reviews.csv",
        limit=None
    )