
from pathlib import Path
import re

from datasets import load_dataset

from sklearn.model_selection import train_test_split


DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "bitext_customer_support.csv"


def download_raw_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split="train")
    dataframe = dataset.to_pandas()

    dataframe.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8")

    print(f"Dataset saved to: {RAW_DATA_PATH}")
    print(f"Number of rows: {len(dataframe)}")
    print(f"Columns: {list(dataframe.columns)}")

    return dataframe



def inspect_raw_dataset(dataframe):
    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\n=== COLUMN DATA TYPES ===")
    print(dataframe.dtypes)

    print("\n=== MISSING VALUES ===")
    print(dataframe.isnull().sum())

    print("\n=== CATEGORY DISTRIBUTION ===")
    print(dataframe["category"].value_counts())

    print("\n=== TOP 15 INTENTS ===")
    print(dataframe["intent"].value_counts().head(15))

    print("\n=== UNIQUE LABEL COUNTS ===")
    print(f"Unique categories: {dataframe['category'].nunique()}")
    print(f"Unique intents: {dataframe['intent'].nunique()}")

    print("\n=== SAMPLE ROWS ===")
    print(
        dataframe[["instruction", "category", "intent", "response"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n=== DUPLICATE COUNTS ===")
    print(
        "Duplicate instructions:",
        dataframe["instruction"].duplicated().sum(),
    )



def normalize_prompt(text):
    """Normalize capitalization and whitespace for prompt comparisons."""
    text = str(text).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def audit_duplicate_prompts(dataframe):
    audit_df = dataframe.copy()

    # Create a comparison key without modifying the original instructions.
    audit_df["normalized_instruction"] = audit_df["instruction"].map(
        normalize_prompt
    )

    duplicate_mask = audit_df["normalized_instruction"].duplicated(
        keep=False
    )
    duplicate_rows = audit_df[duplicate_mask]

    # Count repeated rows beyond the first occurrence of each prompt.
    repeated_rows = audit_df["normalized_instruction"].duplicated().sum()

    # Check whether one normalized prompt has multiple label combinations.
    label_counts = (
        duplicate_rows.groupby("normalized_instruction")[
            ["category", "intent"]
        ]
        .apply(lambda group: len(group.drop_duplicates()))
    )

    conflicting_prompts = label_counts[label_counts > 1]

    print("\n=== NORMALIZED DUPLICATE AUDIT ===")
    print(f"Rows involved in duplicate groups: {len(duplicate_rows)}")
    print(f"Repeated rows beyond first occurrence: {repeated_rows}")
    print(f"Duplicate prompt groups: {len(label_counts)}")
    print(f"Groups with conflicting label combinations: "
          f"{len(conflicting_prompts)}")

    if not conflicting_prompts.empty:
        print("\n=== CONFLICTING LABEL EXAMPLES ===")
        conflict_examples = (
            duplicate_rows[
                duplicate_rows["normalized_instruction"].isin(
                    conflicting_prompts.index
                )
            ][["instruction", "category", "intent"]]
            .sort_values(["instruction", "category", "intent"])
            .head(20)
        )
        print(conflict_examples.to_string(index=False))
    else:
        print("\nNo conflicting label combinations found in duplicate groups.")


def deduplicate_dataset(dataframe):
    """Remove repeated prompts and save the cleaned dataset."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df = dataframe.copy()

    # Normalize prompts so capitalization and extra whitespace do not
    # cause duplicate prompts to be treated as different examples.
    cleaned_df["normalized_instruction"] = cleaned_df["instruction"].map(
        normalize_prompt
    )

    original_rows = len(cleaned_df)

    # Keep the first occurrence of each normalized prompt.
    cleaned_df = cleaned_df.drop_duplicates(
        subset="normalized_instruction",
        keep="first",
    ).reset_index(drop=True)

    removed_rows = original_rows - len(cleaned_df)

    # Confirm that no duplicate normalized prompts remain.
    remaining_duplicates = cleaned_df["normalized_instruction"].duplicated().sum()

    if remaining_duplicates != 0:
        raise ValueError(
            f"Deduplication failed: {remaining_duplicates} duplicate prompts remain."
        )

    output_path = processed_dir / "bitext_customer_support_deduplicated.csv"
    cleaned_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n=== DEDUPLICATION RESULTS ===")
    print(f"Rows before: {original_rows}")
    print(f"Rows after: {len(cleaned_df)}")
    print(f"Rows removed: {removed_rows}")
    print(f"Remaining duplicate prompts: {remaining_duplicates}")
    print(f"Saved to: {output_path}")

    return cleaned_df




def split_dataset(cleaned_df):
    """Create reproducible, stratified train, validation, and test splits."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataframe = cleaned_df.copy()

    # Ensure the normalized prompt key exists.
    if "normalized_instruction" not in dataframe.columns:
        dataframe["normalized_instruction"] = dataframe["instruction"].map(
            normalize_prompt
        )

    # Reject duplicate prompts before splitting to prevent leakage.
    if dataframe["normalized_instruction"].duplicated().any():
        raise ValueError(
            "Duplicate normalized prompts found. Deduplicate before splitting."
        )

    # Each intent should consistently belong to one category.
    categories_per_intent = dataframe.groupby("intent")["category"].nunique()
    if (categories_per_intent > 1).any():
        raise ValueError("At least one intent maps to multiple categories.")

    # First: reserve 20% for validation and test combined.
    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=dataframe["intent"],
    )

    # Second: divide that 20% equally into validation and test.
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=42,
        stratify=temporary_df["intent"], #preserve each intent's approximate portion in both subsets
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    # Verify that no normalized prompt occurs in multiple splits.
    prompt_sets = {
        name: set(split["normalized_instruction"])
        for name, split in splits.items()
    }

    if prompt_sets["train"] & prompt_sets["validation"]:
        raise ValueError("Prompt leakage detected between train and validation.")

    if prompt_sets["train"] & prompt_sets["test"]:
        raise ValueError("Prompt leakage detected between train and test.")

    if prompt_sets["validation"] & prompt_sets["test"]:
        raise ValueError("Prompt leakage detected between validation and test.")

    # Verify that every split contains all intents.
    all_intents = set(dataframe["intent"].unique())

    for name, split in splits.items():
        missing_intents = all_intents - set(split["intent"].unique())

        if missing_intents:
            raise ValueError(
                f"{name} split is missing intents: {sorted(missing_intents)}"
            )

    # Save the three datasets.
    for name, split in splits.items():
        output_path = processed_dir / f"{name}.csv"
        split.to_csv(output_path, index=False, encoding="utf-8")

    print("\n=== DATA SPLIT RESULTS ===")
    print(f"Total cleaned rows: {len(dataframe)}")

    for name, split in splits.items():
        print(f"\n{name.upper()}")
        print(f"Rows: {len(split)}")
        print(f"Percentage: {len(split) / len(dataframe) * 100:.2f}%")
        print(f"Unique categories: {split['category'].nunique()}")
        print(f"Unique intents: {split['intent'].nunique()}")
        print(f"Saved to: {processed_dir / f'{name}.csv'}")

    print("\nPrompt leakage check: PASSED")
    print("Intent coverage check: PASSED")

    return splits




if __name__ == "__main__":
    dataframe = download_raw_dataset()
    inspect_raw_dataset(dataframe)
    audit_duplicate_prompts(dataframe)

    cleaned_df = deduplicate_dataset(dataframe)
    split_dataset(cleaned_df)