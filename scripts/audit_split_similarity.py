
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "split_similarity_audit.csv"

REQUIRED_COLUMNS = {"instruction", "category", "intent"}


def load_dataset(path: Path, name: str) -> pd.DataFrame:
    """Load a dataset and validate the columns and instruction text."""
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    dataframe = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    if dataframe[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains missing required values.")

    dataframe = dataframe.copy()

    for column in ["instruction", "category", "intent"]:
        dataframe[column] = dataframe[column].astype(str).str.strip()

    if (dataframe["instruction"] == "").any():
        raise ValueError(f"{name} contains blank instructions.")

    return dataframe


def main() -> None:
    train_df = load_dataset(TRAIN_PATH, "Training dataset")
    test_df = load_dataset(TEST_PATH, "Test dataset")

    print(f"Training examples: {len(train_df):,}")
    print(f"Test examples: {len(test_df):,}")

    # Character n-grams capture similar wording and tolerate small
    # spelling differences between instructions.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        dtype=float,
    )

    print("\nFitting character n-gram TF-IDF on training instructions...")

    train_vectors = vectorizer.fit_transform(train_df["instruction"])
    test_vectors = vectorizer.transform(test_df["instruction"])

    print(f"Training matrix shape: {train_vectors.shape}")
    print(f"Test matrix shape: {test_vectors.shape}")

    # Find the single most similar training instruction for each test row.
    nearest_neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nearest_neighbors.fit(train_vectors)

    distances, indices = nearest_neighbors.kneighbors(test_vectors)

    similarities = 1.0 - distances.ravel()
    nearest_indices = indices.ravel()

    nearest_train = train_df.iloc[nearest_indices].reset_index(drop=True)

    audit_df = pd.DataFrame(
        {
            "test_instruction": test_df["instruction"].values,
            "true_category": test_df["category"].values,
            "true_intent": test_df["intent"].values,
            "nearest_train_instruction": nearest_train["instruction"].values,
            "nearest_train_category": nearest_train["category"].values,
            "nearest_train_intent": nearest_train["intent"].values,
            "similarity": similarities,
        }
    )

    audit_df["nearest_category_matches"] = (
        audit_df["true_category"].str.upper()
        == audit_df["nearest_train_category"].str.upper()
    )

    audit_df["nearest_intent_matches"] = (
        audit_df["true_intent"].str.upper()
        == audit_df["nearest_train_intent"].str.upper()
    )

    audit_df = audit_df.sort_values(
        "similarity",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSIMILARITY SUMMARY")
    print("------------------")
    print(f"Mean similarity:   {audit_df['similarity'].mean():.4f}")
    print(f"Median similarity: {audit_df['similarity'].median():.4f}")
    print(f"90th percentile:   {audit_df['similarity'].quantile(0.90):.4f}")
    print(f"95th percentile:   {audit_df['similarity'].quantile(0.95):.4f}")
    print(f"99th percentile:   {audit_df['similarity'].quantile(0.99):.4f}")

    print("\nTEST EXAMPLES ABOVE SIMILARITY THRESHOLDS")
    print("-----------------------------------------")

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        count = int((audit_df["similarity"] >= threshold).sum())
        percentage = count / len(audit_df) * 100

        print(
            f"Similarity >= {threshold:.2f}: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\nNEAREST TRAINING LABEL AGREEMENT")
    print("--------------------------------")
    print(
        "Category agreement: "
        f"{audit_df['nearest_category_matches'].mean() * 100:.2f}%"
    )
    print(
        "Intent agreement:   "
        f"{audit_df['nearest_intent_matches'].mean() * 100:.2f}%"
    )

    print("\nTOP 20 MOST SIMILAR TRAIN/TEST PAIRS")
    print("-----------------------------------")

    for index, row in audit_df.head(20).iterrows():
        print(f"\nPair {index + 1} | Similarity: {row['similarity']:.4f}")
        print(f"TEST:  {row['test_instruction']}")
        print(
            f"       True labels: "
            f"{row['true_category']} / {row['true_intent']}"
        )
        print(f"TRAIN: {row['nearest_train_instruction']}")
        print(
            f"       Train labels: "
            f"{row['nearest_train_category']} / "
            f"{row['nearest_train_intent']}"
        )

    print(f"\nFull audit saved to: {OUTPUT_PATH}")
    print("\nSimilarity audit completed successfully.")


if __name__ == "__main__":
    main()
