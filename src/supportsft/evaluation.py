
import json
import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


REQUIRED_OUTPUT_KEYS = {"category", "intent"}

REQUIRED_EVALUATION_COLUMNS = {
    "instruction",
    "category",
    "intent",
    "raw_output",
}

UNKNOWN_LABEL = "__INVALID_OR_UNKNOWN__"


def parse_model_output(raw_output: str) -> dict:
    """Parse raw model output into a JSON dictionary."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return {}

    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed


def is_valid_json(raw_output: str) -> bool:
    """Check JSON syntax independently of the expected output schema."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return False

    text = raw_output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def validate_prediction(parsed_output: dict) -> bool:
    """Check whether the output has the required JSON structure."""
    if not isinstance(parsed_output, dict):
        return False

    if not REQUIRED_OUTPUT_KEYS.issubset(parsed_output.keys()):
        return False

    category = parsed_output["category"]
    intent = parsed_output["intent"]

    if not isinstance(category, str) or not category.strip():
        return False

    if not isinstance(intent, str) or not intent.strip():
        return False

    return True


def normalize_prediction_label(value) -> str:
    """Normalize a classification label for consistent comparison."""
    if not isinstance(value, str):
        return ""

    return value.strip().upper()


def prepare_predictions(
    dataframe: pd.DataFrame,
    category_labels: list[str] | None = None,
    intent_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse raw outputs and prepare predictions for evaluation.

    Unknown and invalid labels receive a dedicated sentinel value
    in the evaluation prediction columns.
    """
    missing_columns = REQUIRED_EVALUATION_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Cannot prepare predictions from an empty dataset.")

    results = dataframe.copy()

    results["true_category"] = results["category"].apply(
        normalize_prediction_label
    )

    results["true_intent"] = results["intent"].apply(
        normalize_prediction_label
    )

    if category_labels is None:
        category_labels = sorted(results["true_category"].unique())

    if intent_labels is None:
        intent_labels = sorted(results["true_intent"].unique())

    category_labels = {
        normalize_prediction_label(label) for label in category_labels
    }

    intent_labels = {
        normalize_prediction_label(label) for label in intent_labels
    }

    if "" in category_labels or "" in intent_labels:
        raise ValueError("Ground-truth label vocabularies cannot contain blanks.")

    if UNKNOWN_LABEL in category_labels or UNKNOWN_LABEL in intent_labels:
        raise ValueError(
            f"{UNKNOWN_LABEL} is reserved for invalid predictions."
        )

    if not results["true_category"].isin(category_labels).all():
        raise ValueError("Ground-truth data contains an unknown category.")

    if not results["true_intent"].isin(intent_labels).all():
        raise ValueError("Ground-truth data contains an unknown intent.")

    parsed_outputs = results["raw_output"].apply(parse_model_output)

    results["json_valid"] = results["raw_output"].apply(is_valid_json)
    results["schema_valid"] = parsed_outputs.apply(validate_prediction)

    results["predicted_category"] = parsed_outputs.apply(
        lambda output: (
            normalize_prediction_label(output.get("category"))
            if validate_prediction(output)
            else ""
        )
    )

    results["predicted_intent"] = parsed_outputs.apply(
        lambda output: (
            normalize_prediction_label(output.get("intent"))
            if validate_prediction(output)
            else ""
        )
    )

    results["category_label_valid"] = results["predicted_category"].isin(
        category_labels
    )

    results["intent_label_valid"] = results["predicted_intent"].isin(
        intent_labels
    )

    results["evaluation_predicted_category"] = results[
        "predicted_category"
    ].where(
        results["category_label_valid"],
        UNKNOWN_LABEL,
    )

    results["evaluation_predicted_intent"] = results[
        "predicted_intent"
    ].where(
        results["intent_label_valid"],
        UNKNOWN_LABEL,
    )

    results["category_correct"] = (
        results["predicted_category"] == results["true_category"]
    ) & results["category_label_valid"]

    results["intent_correct"] = (
        results["predicted_intent"] == results["true_intent"]
    ) & results["intent_label_valid"]

    results["exact_match"] = (
        results["category_correct"] & results["intent_correct"]
    )

    return results


def calculate_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict:
    """
    Calculate classification metrics over ground-truth classes only.

    Predictions outside the ground-truth vocabulary are mapped to
    UNKNOWN_LABEL. They count as false negatives for their true classes,
    but UNKNOWN_LABEL is excluded from macro and weighted averages.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Ground-truth and prediction lengths must match.")

    if not y_true:
        raise ValueError("Cannot calculate metrics for empty inputs.")

    if any(not isinstance(label, str) or not label for label in y_true):
        raise ValueError("Ground-truth labels must be non-empty strings.")

    known_labels = sorted(set(y_true))

    if UNKNOWN_LABEL in known_labels:
        raise ValueError(
            f"{UNKNOWN_LABEL} is reserved for invalid predictions."
        )

    normalized_predictions = [
        prediction if prediction in known_labels else UNKNOWN_LABEL
        for prediction in y_pred
    ]

    all_labels = known_labels + [UNKNOWN_LABEL]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        normalized_predictions,
        labels=all_labels,
        average=None,
        zero_division=0,
    )

    # Exclude the final UNKNOWN_LABEL entry from the averages.
    known_precision = precision[:-1]
    known_recall = recall[:-1]
    known_f1 = f1[:-1]
    known_support = support[:-1]

    total_support = known_support.sum()

    macro_precision = known_precision.mean()
    macro_recall = known_recall.mean()
    macro_f1 = known_f1.mean()

    weighted_precision = (
        (known_precision * known_support).sum() / total_support
    )
    weighted_recall = (
        (known_recall * known_support).sum() / total_support
    )
    weighted_f1 = (
        (known_f1 * known_support).sum() / total_support
    )

    return {
        "accuracy": accuracy_score(y_true, normalized_predictions),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "unknown_prediction_count": sum(
            prediction == UNKNOWN_LABEL
            for prediction in normalized_predictions
        ),
    }


def calculate_evaluation_metrics(
    prepared_df: pd.DataFrame,
) -> dict:
    """Calculate overall JSON, classification and exact-match metrics."""
    if prepared_df.empty:
        raise ValueError("Cannot evaluate an empty dataset.")

    category_metrics = calculate_classification_metrics(
        prepared_df["true_category"].tolist(),
        prepared_df["evaluation_predicted_category"].tolist(),
    )

    intent_metrics = calculate_classification_metrics(
        prepared_df["true_intent"].tolist(),
        prepared_df["evaluation_predicted_intent"].tolist(),
    )

    total_examples = len(prepared_df)

    return {
        "total_examples": total_examples,
        "json_validity_rate": prepared_df["json_valid"].mean(),
        "schema_validity_rate": prepared_df["schema_valid"].mean(),
        "category_label_validity_rate": prepared_df[
            "category_label_valid"
        ].mean(),
        "intent_label_validity_rate": prepared_df[
            "intent_label_valid"
        ].mean(),
        "category_unknown_prediction_count": (
            category_metrics["unknown_prediction_count"]
        ),
        "intent_unknown_prediction_count": (
            intent_metrics["unknown_prediction_count"]
        ),
        "category_unknown_prediction_rate": (
            category_metrics["unknown_prediction_count"] / total_examples
        ),
        "intent_unknown_prediction_rate": (
            intent_metrics["unknown_prediction_count"] / total_examples
        ),
        "category_accuracy": category_metrics["accuracy"],
        "category_macro_precision": category_metrics["macro_precision"],
        "category_macro_recall": category_metrics["macro_recall"],
        "category_macro_f1": category_metrics["macro_f1"],
        "category_weighted_precision": category_metrics["weighted_precision"],
        "category_weighted_recall": category_metrics["weighted_recall"],
        "category_weighted_f1": category_metrics["weighted_f1"],
        "intent_accuracy": intent_metrics["accuracy"],
        "intent_macro_precision": intent_metrics["macro_precision"],
        "intent_macro_recall": intent_metrics["macro_recall"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "intent_weighted_precision": intent_metrics["weighted_precision"],
        "intent_weighted_recall": intent_metrics["weighted_recall"],
        "intent_weighted_f1": intent_metrics["weighted_f1"],
        "exact_match_accuracy": prepared_df["exact_match"].mean(),
    }


def generate_classification_reports(
    prepared_df: pd.DataFrame,
) -> dict:
    """
    Generate per-class reports.

    The UNKNOWN_LABEL row is retained for diagnostics. Macro and weighted
    averages are recalculated using only ground-truth classes.
    """
    if prepared_df.empty:
        raise ValueError("Cannot generate reports for an empty dataset.")

    def build_report(
        true_labels: list[str],
        predicted_labels: list[str],
    ) -> dict:
        known_labels = sorted(set(true_labels))

        if UNKNOWN_LABEL in known_labels:
            raise ValueError(
                f"{UNKNOWN_LABEL} is reserved for invalid predictions."
            )

        normalized_predictions = [
            prediction if prediction in known_labels else UNKNOWN_LABEL
            for prediction in predicted_labels
        ]

        all_labels = known_labels + [UNKNOWN_LABEL]

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                true_labels,
                normalized_predictions,
                labels=all_labels,
                average=None,
                zero_division=0,
            )
        )

        report = classification_report(
            true_labels,
            normalized_predictions,
            labels=all_labels,
            output_dict=True,
            zero_division=0,
        )

        # Recalculate aggregate metrics without the diagnostic unknown row.
        known_precision = precision[:-1]
        known_recall = recall[:-1]
        known_f1 = f1[:-1]
        known_support = support[:-1]
        total_support = known_support.sum()

        report["macro avg"] = {
            "precision": known_precision.mean(),
            "recall": known_recall.mean(),
            "f1-score": known_f1.mean(),
            "support": int(total_support),
        }

        report["weighted avg"] = {
            "precision": (
                known_precision * known_support
            ).sum() / total_support,
            "recall": (
                known_recall * known_support
            ).sum() / total_support,
            "f1-score": (
                known_f1 * known_support
            ).sum() / total_support,
            "support": int(total_support),
        }

        return report

    category_report = build_report(
        prepared_df["true_category"].tolist(),
        prepared_df["evaluation_predicted_category"].tolist(),
    )

    intent_report = build_report(
        prepared_df["true_intent"].tolist(),
        prepared_df["evaluation_predicted_intent"].tolist(),
    )

    return {
        "category": category_report,
        "intent": intent_report,
    }
