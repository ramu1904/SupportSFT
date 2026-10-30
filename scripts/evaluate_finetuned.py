
"""Evaluate the fine-tuned model and generate all evaluation reports."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =========================================================
# 1. PROJECT PATHS AND IMPORTS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from supportsft.evaluation import (
    UNKNOWN_LABEL,
    calculate_evaluation_metrics,
    generate_classification_reports,
    prepare_predictions,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "fine_tuned"
    / "fine_tuned_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "benchmark_a"
FINETUNED_DIR = PROJECT_ROOT / "artifacts" / "fine_tuned"

CONFUSION_DIR = REPORTS_DIR / "fine_tuned_confusion_matrices"

EXPECTED_TEST_ROWS = 2428

REQUIRED_COLUMNS = {
    "instruction",
    "category",
    "intent",
}

PREDICTION_COLUMNS = REQUIRED_COLUMNS | {"raw_output"}

ERROR_COLUMNS = [
    "json_valid",
    "schema_valid",
    "category_label_valid",
    "intent_label_valid",
    "category_correct",
    "intent_correct",
]


# =========================================================
# 2. JSON SERIALIZATION
# =========================================================

def make_json_safe(value):
    """Convert NumPy and Pandas values into JSON-compatible values."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if value is None:
        return None

    if pd.isna(value):
        return None

    return value


def save_json(data, output_path):
    """Save evaluation results as formatted JSON."""

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(data),
            file,
            indent=4,
            ensure_ascii=False,
        )


# =========================================================
# 3. LOAD AND VALIDATE DATA
# =========================================================

def load_and_validate_files():
    """Load files and verify prediction-to-test alignment."""

    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        PREDICTIONS_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file does not exist: {path}"
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    if not REQUIRED_COLUMNS.issubset(train_df.columns):
        raise ValueError("Training CSV is missing required columns.")

    if not REQUIRED_COLUMNS.issubset(test_df.columns):
        raise ValueError("Test CSV is missing required columns.")

    if not PREDICTION_COLUMNS.issubset(predictions_df.columns):
        raise ValueError(
            "Prediction CSV must contain: "
            f"{sorted(PREDICTION_COLUMNS)}"
        )

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS} test examples, "
            f"found {len(test_df)}."
        )

    if len(predictions_df) != len(test_df):
        raise ValueError(
            "Prediction count does not match test-set size."
        )

    if train_df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Training data contains missing values.")

    if test_df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Test data contains missing values.")

    if predictions_df[list(PREDICTION_COLUMNS)].isnull().any().any():
        raise ValueError("Prediction data contains missing values.")

    # Verify exact examples, labels, and row ordering.
    comparison_columns = [
        "instruction",
        "category",
        "intent",
    ]

    expected = (
        test_df[comparison_columns]
        .astype(str)
        .reset_index(drop=True)
    )

    actual = (
        predictions_df[comparison_columns]
        .astype(str)
        .reset_index(drop=True)
    )

    if not expected.equals(actual):
        raise ValueError(
            "Predictions do not match the fixed test dataset. "
            "Evaluation stopped."
        )

    print("File validation successful.")
    print(f"Training examples: {len(train_df):,}")
    print(f"Test examples: {len(test_df):,}")
    print(f"Predictions: {len(predictions_df):,}")

    return train_df, predictions_df


# =========================================================
# 4. PRINT METRICS
# =========================================================

def print_metrics(metrics):
    """Print overall evaluation metrics."""

    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 60)

    for name, value in metrics.items():

        if isinstance(value, (float, np.floating)):
            print(f"{name:<42} {value:.6f}")

        elif isinstance(value, (int, np.integer)):
            print(f"{name:<42} {value:,}")

        else:
            print(f"{name:<42} {value}")


# =========================================================
# 5. GENERATE CONFUSION MATRICES
# =========================================================

def generate_confusion_matrix(
    dataframe,
    true_column,
    predicted_column,
    valid_labels,
    matrix_name,
):
    """Generate a confusion matrix, CSV and PNG heatmap."""

    normalized_labels = sorted(
        {
            str(label).strip().upper()
            for label in valid_labels
        }
    )

    all_labels = normalized_labels + [UNKNOWN_LABEL]

    matrix = confusion_matrix(
        dataframe[true_column],
        dataframe[predicted_column],
        labels=all_labels,
    )

    if matrix.sum() != len(dataframe):
        raise ValueError(
            f"{matrix_name} confusion matrix contains "
            f"{matrix.sum()} examples, but expected {len(dataframe)}."
        )

    # Save numerical confusion matrix.
    matrix_df = pd.DataFrame(
        matrix,
        index=all_labels,
        columns=all_labels,
    )

    matrix_df.index.name = "actual_label"
    matrix_df.columns.name = "predicted_label"

    csv_path = (
        CONFUSION_DIR
        / f"{matrix_name}_confusion_matrix.csv"
    )

    matrix_df.to_csv(csv_path, encoding="utf-8")

    # Generate heatmap.
    number_of_labels = len(all_labels)

    figure_size = max(10, number_of_labels * 0.65)

    fig, ax = plt.subplots(
        figsize=(figure_size, figure_size * 0.85)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of predictions",
    )

    display_labels = [
        "UNKNOWN" if label == UNKNOWN_LABEL else label
        for label in all_labels
    ]

    positions = np.arange(number_of_labels)

    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(
        display_labels,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticklabels(
        display_labels,
        fontsize=8,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_title(
        f"Fine-Tuned {matrix_name.title()} Confusion Matrix"
    )

    # Annotate non-zero cells.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):

            value = matrix[row, column]

            if value > 0:
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()

    png_path = (
        CONFUSION_DIR
        / f"{matrix_name}_confusion_matrix.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\n{matrix_name.upper()} CONFUSION MATRIX")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Examples represented: {matrix.sum():,}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


# =========================================================
# 6. MAIN EVALUATION PIPELINE
# =========================================================

def main():
    """Run fine-tuned evaluation and generate all reports."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and validate files.
    train_df, predictions_df = load_and_validate_files()

    # Step 2: Get valid labels from training data.
    category_labels = sorted(
        train_df["category"].astype(str).unique().tolist()
    )

    intent_labels = sorted(
        train_df["intent"].astype(str).unique().tolist()
    )

    print(f"\nValid categories: {len(category_labels)}")
    print(f"Valid intents: {len(intent_labels)}")

    # Step 3: Prepare predictions.
    prepared_df = prepare_predictions(
        predictions_df,
        category_labels=category_labels,
        intent_labels=intent_labels,
    )

    # Step 4: Calculate overall metrics.
    metrics = calculate_evaluation_metrics(prepared_df)

    # Step 5: Generate classification reports.
    classification_reports = generate_classification_reports(
        prepared_df
    )

    # Step 6: Save prepared predictions.
    prepared_path = (
        FINETUNED_DIR / "fine_tuned_prepared_predictions.csv"
    )

    prepared_df.to_csv(
        prepared_path,
        index=False,
        encoding="utf-8",
    )

    # Step 7: Save incorrect and invalid predictions.
    missing_columns = (
        set(ERROR_COLUMNS) - set(prepared_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing evaluation columns: "
            f"{sorted(missing_columns)}"
        )

    error_mask = ~prepared_df[ERROR_COLUMNS].all(axis=1)

    error_df = prepared_df.loc[error_mask].copy()

    error_path = (
        REPORTS_DIR / "fine_tuned_error_analysis.csv"
    )

    error_df.to_csv(
        error_path,
        index=False,
        encoding="utf-8",
    )

    # Step 8: Save overall metrics.
    metrics_path = (
        REPORTS_DIR / "fine_tuned_metrics.json"
    )

    save_json(metrics, metrics_path)

    # Step 9: Save classification reports.
    reports_path = (
        REPORTS_DIR / "fine_tuned_classification_reports.json"
    )

    save_json(classification_reports, reports_path)

    # Step 10: Category confusion matrix.
    generate_confusion_matrix(
        dataframe=prepared_df,
        true_column="true_category",
        predicted_column="evaluation_predicted_category",
        valid_labels=category_labels,
        matrix_name="category",
    )

    # Step 11: Intent confusion matrix.
    generate_confusion_matrix(
        dataframe=prepared_df,
        true_column="true_intent",
        predicted_column="evaluation_predicted_intent",
        valid_labels=intent_labels,
        matrix_name="intent",
    )

    # Step 12: Print metrics.
    print_metrics(metrics)

    # Step 13: Print saved file locations.
    print("\n" + "=" * 60)
    print("ALL FINE-TUNED EVALUATION FILES SAVED")
    print("=" * 60)

    print(f"Metrics JSON: {metrics_path}")
    print(f"Classification reports: {reports_path}")
    print(f"Error analysis CSV: {error_path}")
    print(f"Prepared predictions CSV: {prepared_path}")
    print(f"Confusion matrices folder: {CONFUSION_DIR}")

    print(
        f"\nIncorrect or invalid predictions: "
        f"{len(error_df):,} / {len(prepared_df):,}"
    )

    print("\nFine-tuned evaluation completed successfully.")


if __name__ == "__main__":
    main()
