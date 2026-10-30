
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

SPLIT_FILES = {
    "train": ROOT_DIR / "data" / "processed" / "train.csv",
    "validation": ROOT_DIR / "data" / "processed" / "validation.csv",
    "test": ROOT_DIR / "data" / "processed" / "test.csv",
}

REPORT_DIR = ROOT_DIR / "reports"

TAG_MEANINGS = {
    "M": "Morphological variation",
    "L": "Lexical/semantic variation",
    "B": "Basic syntactic structure",
    "I": "Interrogative structure",
    "C": "Coordinated syntactic structure",
    "N": "Negation",
    "P": "Politeness",
    "Q": "Colloquial language",
    "W": "Offensive language",
    "K": "Keyword mode",
    "E": "Abbreviations",
    "Z": "Spelling errors and typos",
}

UNUSED_DOCUMENTED_TAGS = {"D", "G", "R", "Y"}


def load_splits():
    """Load the existing train, validation, and test CSV files."""
    dataframes = {}

    for split_name, file_path in SPLIT_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {split_name} file: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        if "flags" not in dataframe.columns:
            raise ValueError(
                f"The {split_name} dataset has no 'flags' column."
            )

        dataframes[split_name] = dataframe

    return dataframes


def normalize_flags(value):
    """Normalize a concatenated flags string without changing its meaning."""
    if pd.isna(value):
        return ""

    return "".join(str(value).strip().upper().split())


def audit_splits(dataframes):
    """Count individual linguistic tags and complete flag combinations."""
    tag_rows = []
    combination_rows = []
    unknown_tag_rows = []

    for split_name, dataframe in dataframes.items():
        flags_series = dataframe["flags"].map(normalize_flags)

        tag_counts = Counter()
        combination_counts = Counter()
        unknown_counts = Counter()

        for flags in flags_series:
            # Each documented tag is represented by one letter.
            present_tags = set(flags)

            for tag in present_tags:
                if tag in TAG_MEANINGS:
                    tag_counts[tag] += 1
                else:
                    unknown_counts[tag] += 1

            combination_counts[flags or "<EMPTY>"] += 1

        total_rows = len(dataframe)

        print(f"\n{'=' * 65}")
        print(f"SPLIT: {split_name.upper()}")
        print(f"{'=' * 65}")
        print(f"Total examples: {total_rows}")
        print(f"Unique flag combinations: {len(combination_counts)}")

        print("\nINDIVIDUAL TAG COUNTS")
        print("-" * 65)

        for tag, meaning in TAG_MEANINGS.items():
            count = tag_counts[tag]
            percentage = count / total_rows * 100 if total_rows else 0

            print(
                f"{tag} | {meaning:<32} "
                f"{count:>7} ({percentage:6.2f}%)"
            )

            tag_rows.append(
                {
                    "split": split_name,
                    "tag": tag,
                    "meaning": meaning,
                    "count": count,
                    "percentage": percentage,
                }
            )

        print("\nMOST COMMON FLAG COMBINATIONS")
        print("-" * 65)

        for flags, count in combination_counts.most_common(20):
            percentage = count / total_rows * 100

            print(
                f"{flags:<15} {count:>7} "
                f"({percentage:6.2f}%)"
            )

            combination_rows.append(
                {
                    "split": split_name,
                    "flags": flags,
                    "count": count,
                    "percentage": percentage,
                }
            )

        for tag, count in unknown_counts.items():
            unknown_tag_rows.append(
                {
                    "split": split_name,
                    "tag": tag,
                    "count": count,
                }
            )

        if unknown_counts:
            print("\nUNDOCUMENTED OR UNEXPECTED FLAG CHARACTERS")
            print(dict(unknown_counts))

    return tag_rows, combination_rows, unknown_tag_rows


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataframes = load_splits()

    tag_rows, combination_rows, unknown_rows = audit_splits(
        dataframes
    )

    tag_report = pd.DataFrame(tag_rows)
    combination_report = pd.DataFrame(combination_rows)
    unknown_report = pd.DataFrame(
        unknown_rows,
        columns=["split", "tag", "count"],
    )

    tag_path = REPORT_DIR / "language_tag_audit.csv"
    combination_path = REPORT_DIR / "language_tag_combinations.csv"
    unknown_path = REPORT_DIR / "unknown_language_tags.csv"

    tag_report.to_csv(tag_path, index=False)
    combination_report.to_csv(combination_path, index=False)
    unknown_report.to_csv(unknown_path, index=False)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)
    print(f"Tag report: {tag_path}")
    print(f"Combination report: {combination_path}")
    print(f"Unexpected tag report: {unknown_path}")


if __name__ == "__main__":
    main()
