from dotenv import load_dotenv
import os
from groq import Groq

# load .env
load_dotenv()

# ambil API key
api_key = os.getenv("GROQ_API_KEY")

print("API KEY:", api_key[:10], "...")  # biar aman, tidak tampil full

# init client
client = Groq(api_key=api_key)

# test request
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Halo, ini test koneksi Groq"}
    ],
    temperature=0
)

print("\nResponse dari Groq:")
print(response.choices[0].message.content)