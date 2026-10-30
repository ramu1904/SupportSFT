
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

SPLIT_FILES = {
    "train": ROOT_DIR / "data" / "processed" / "train.csv",
    "validation": ROOT_DIR / "data" / "processed" / "validation.csv",
    "test": ROOT_DIR / "data" / "processed" / "test.csv",
}

REPORT_DIR = ROOT_DIR / "reports"

CANDIDATE_TAGS = ["Z", "Q", "N", "P", "W"]
UNKNOWN_TAGS = ["S", "V"]


def load_all_splits():
    """Load all existing splits and combine them for the audit."""
    frames = []

    for split_name, file_path in SPLIT_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        dataframe = pd.read_csv(file_path)

        required = {"flags", "category", "intent", "instruction"}
        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{split_name} is missing columns: {sorted(missing)}"
            )

        dataframe = dataframe.copy()
        dataframe["split"] = split_name
        dataframe["flags_normalized"] = (
            dataframe["flags"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace(" ", "", regex=False)
        )

        frames.append(dataframe)

    return pd.concat(frames, ignore_index=True)


def contains_tag(flags, tag):
    """Check whether a flag string contains a particular tag."""
    return tag in set(flags)


def audit_candidate_tags(dataframe):
    """Measure example counts and label coverage for each candidate tag."""
    summary_rows = []
    category_rows = []
    intent_rows = []

    for tag in CANDIDATE_TAGS:
        tagged = dataframe[
            dataframe["flags_normalized"].map(
                lambda flags: contains_tag(flags, tag)
            )
        ]

        print(f"\n{'=' * 65}")
        print(f"TAG: {tag}")
        print(f"{'=' * 65}")
        print(f"Examples containing tag: {len(tagged)}")

        for split_name in ["train", "validation", "test"]:
            split_tagged = tagged[tagged["split"] == split_name]

            print(
                f"{split_name:<12}: "
                f"{len(split_tagged):>6} examples"
            )

        category_count = tagged["category"].nunique()
        intent_count = tagged["intent"].nunique()

        print(f"Categories represented: {category_count}")
        print(f"Intents represented: {intent_count}")

        summary_rows.append(
            {
                "tag": tag,
                "total_examples": len(tagged),
                "train_examples": int(
                    (tagged["split"] == "train").sum()
                ),
                "validation_examples": int(
                    (tagged["split"] == "validation").sum()
                ),
                "test_examples": int(
                    (tagged["split"] == "test").sum()
                ),
                "categories_represented": category_count,
                "intents_represented": intent_count,
            }
        )

        category_counts = (
            tagged.groupby("category")
            .size()
            .sort_values(ascending=False)
        )

        intent_counts = (
            tagged.groupby("intent")
            .size()
            .sort_values(ascending=False)
        )

        for category, count in category_counts.items():
            category_rows.append(
                {
                    "tag": tag,
                    "category": category,
                    "count": int(count),
                }
            )

        for intent, count in intent_counts.items():
            intent_rows.append(
                {
                    "tag": tag,
                    "intent": intent,
                    "count": int(count),
                }
            )

        print("\nCategory counts:")
        print(category_counts.to_string())

        print("\nIntent counts:")
        print(intent_counts.to_string())


def audit_unknown_tags(dataframe):
    """Inspect undocumented S and V flags without assigning meanings."""
    print("\n" + "=" * 65)
    print("UNDOCUMENTED FLAG INSPECTION")
    print("=" * 65)

    for tag in UNKNOWN_TAGS:
        matching = dataframe[
            dataframe["flags_normalized"].map(
                lambda flags: contains_tag(flags, tag)
            )
        ]

        print(f"\nFlag: {tag}")
        print(f"Total examples: {len(matching)}")
        print("Flag combinations:")
        print(
            matching["flags_normalized"]
            .value_counts()
            .head(20)
            .to_string()
        )

        print("\nExample instructions:")
        for instruction in matching["instruction"].head(10):
            print(f"- {instruction}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_all_splits()

    audit_candidate_tags(dataframe)
    audit_unknown_tags(dataframe)

    summary_rows = []
    category_rows = []
    intent_rows = []

    for tag in CANDIDATE_TAGS:
        tagged = dataframe[
            dataframe["flags_normalized"].map(
                lambda flags: contains_tag(flags, tag)
            )
        ]

        summary_rows.append(
            {
                "tag": tag,
                "total_examples": len(tagged),
                "train_examples": int(
                    (tagged["split"] == "train").sum()
                ),
                "validation_examples": int(
                    (tagged["split"] == "validation").sum()
                ),
                "test_examples": int(
                    (tagged["split"] == "test").sum()
                ),
                "categories_represented": tagged["category"].nunique(),
                "intents_represented": tagged["intent"].nunique(),
            }
        )

        for category, count in tagged["category"].value_counts().items():
            category_rows.append(
                {"tag": tag, "category": category, "count": int(count)}
            )

        for intent, count in tagged["intent"].value_counts().items():
            intent_rows.append(
                {"tag": tag, "intent": intent, "count": int(count)}
            )

    pd.DataFrame(summary_rows).to_csv(
        REPORT_DIR / "tag_coverage_summary.csv", index=False
    )
    pd.DataFrame(category_rows).to_csv(
        REPORT_DIR / "tag_category_coverage.csv", index=False
    )
    pd.DataFrame(intent_rows).to_csv(
        REPORT_DIR / "tag_intent_coverage.csv", index=False
    )

    print("\nCoverage reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
