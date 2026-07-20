import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_DIR = os.path.join("data")

INPUT_FILE = os.path.join(
    DATA_DIR,
    "cleaned_reviews_20.csv"
)

UNLABELED_DIR = os.path.join(
    DATA_DIR,
    "unlabeled"
)

# Load data
df = pd.read_csv(INPUT_FILE)

print(f"Total dataset: {len(df)}")

# Check distribution by version
print("\nDistribusi review per versi:")
print(df['reviewCreatedVersion'].value_counts())

# Split 1
# 80% TRAIN
# 10% VALIDATION
# 10% TEST
# Stratified by version
# =========================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=df['reviewCreatedVersion']
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=temp_df['reviewCreatedVersion']
)

# Split 2
# Train Set:
# 80% Pseudo-label
# 20% Manual annotation
# =========================================================
pseudo_df, manual_train_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=train_df['reviewCreatedVersion']
)


# Info
print("\n===== HASIL SPLIT DATASET =====")
print(f"Train Set Total         : {len(train_df)}")
print(f"Pseudo-label Train Set  : {len(pseudo_df)}")
print(f"Manual Train Set        : {len(manual_train_df)}")
print(f"Validation Set          : {len(validation_df)}")
print(f"Test Set                : {len(test_df)}")


# Check distribution by version
print("\n====== DISTRIBUSI VERSI ======")
print("Train Set:")
print(train_df['reviewCreatedVersion'].value_counts(normalize=True))

print("\nValidation Set:")
print(validation_df['reviewCreatedVersion'].value_counts(normalize=True))

print("\nTest Set:")
print(test_df['reviewCreatedVersion'].value_counts(normalize=True))

# Save dataset
os.makedirs(
    UNLABELED_DIR,
    exist_ok=True
)

train_df.to_csv(
    os.path.join(
        UNLABELED_DIR,
        "train.csv"
    ),
    index=False,
    encoding="utf-8"
)

pseudo_df.to_csv(
    os.path.join(
        UNLABELED_DIR,
        "train_pseudo.csv"
    ),
    index=False,
    encoding="utf-8"
)

manual_train_df.to_csv(
    os.path.join(
        UNLABELED_DIR,
        "train_manual.csv"
    ),
    index=False,
    encoding="utf-8"
)

validation_df.to_csv(
    os.path.join(
        UNLABELED_DIR,
        "validation.csv"
    ),
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    os.path.join(
        UNLABELED_DIR,
        "test.csv"
    ),
    index=False,
    encoding="utf-8"
)

print(f"\nDataset berhasil disimpan di folder: {UNLABELED_DIR}")