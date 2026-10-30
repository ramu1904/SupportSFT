from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmark_a"

BASELINE_FILE = REPORT_DIR / "baseline_metrics.json"
FINE_TUNED_FILE = REPORT_DIR / "fine_tuned_metrics.json"

OUTPUT_CSV = REPORT_DIR / "model_comparison.csv"
OUTPUT_JSON = REPORT_DIR / "model_comparison.json"
OUTPUT_SUMMARY = REPORT_DIR / "comparison_summary.md"


def load_metrics(path: Path) -> dict:
    """Load an evaluation metrics JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_rate_metric(metric_name: str) -> bool:
    """Identify metrics represented as proportions between 0 and 1."""
    return (
        metric_name.endswith("_rate")
        or metric_name.endswith("_accuracy")
        or "_precision" in metric_name
        or "_recall" in metric_name
        or "_f1" in metric_name
    )


def build_comparison(
    baseline: dict,
    fine_tuned: dict,
) -> list[dict]:
    """Build a side-by-side comparison of all shared metrics."""
    if baseline["total_examples"] != fine_tuned["total_examples"]:
        raise ValueError(
            "Baseline and fine-tuned evaluations use different "
            "numbers of test examples."
        )

    rows = []

    for metric_name in sorted(set(baseline) & set(fine_tuned)):
        baseline_value = baseline[metric_name]
        fine_tuned_value = fine_tuned[metric_name]

        if not isinstance(baseline_value, (int, float)):
            continue

        if not isinstance(fine_tuned_value, (int, float)):
            continue

        difference = fine_tuned_value - baseline_value

        row = {
            "metric": metric_name,
            "baseline_value": baseline_value,
            "fine_tuned_value": fine_tuned_value,
            "absolute_difference": difference,
        }

        if is_rate_metric(metric_name):
            row["baseline_percent"] = baseline_value * 100
            row["fine_tuned_percent"] = fine_tuned_value * 100
            row["change_percentage_points"] = difference * 100
        else:
            row["baseline_percent"] = ""
            row["fine_tuned_percent"] = ""
            row["change_percentage_points"] = ""

        rows.append(row)

    return rows


def save_comparison_csv(rows: list[dict]) -> None:
    """Save all comparison metrics in CSV format."""
    fieldnames = [
        "metric",
        "baseline_value",
        "fine_tuned_value",
        "absolute_difference",
        "baseline_percent",
        "fine_tuned_percent",
        "change_percentage_points",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_json(
    baseline: dict,
    fine_tuned: dict,
    rows: list[dict],
) -> None:
    """Save the complete comparison in structured JSON format."""
    result = {
        "project": "SupportSFT",
        "evaluation": {
            "test_examples": baseline["total_examples"],
            "baseline": baseline,
            "fine_tuned": fine_tuned,
        },
        "comparison": rows,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)


def save_bar_chart(
    filename: str,
    title: str,
    metric_names: list[str],
    display_names: list[str],
    baseline: dict,
    fine_tuned: dict,
) -> None:
    """Create a baseline-versus-fine-tuned percentage bar chart."""
    baseline_values = [
        baseline[name] * 100 for name in metric_names
    ]
    fine_tuned_values = [
        fine_tuned[name] * 100 for name in metric_names
    ]

    positions = list(range(len(metric_names)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    baseline_bars = ax.bar(
        [position - width / 2 for position in positions],
        baseline_values,
        width,
        label="Baseline",
    )

    fine_tuned_bars = ax.bar(
        [position + width / 2 for position in positions],
        fine_tuned_values,
        width,
        label="Fine-tuned",
    )

    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax.bar_label(
        baseline_bars,
        labels=[f"{value:.2f}%" for value in baseline_values],
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        fine_tuned_bars,
        labels=[f"{value:.2f}%" for value in fine_tuned_values],
        padding=3,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    baseline: dict,
    fine_tuned: dict,
) -> None:
    """Write a concise human-readable comparison summary."""
    test_count = baseline["total_examples"]

    summary = f"""# SupportSFT: Baseline vs Fine-tuned Model

## Evaluation setup

Both models were evaluated on the same **{test_count:,} test examples**.

- Baseline: Original Qwen2.5-0.5B-Instruct model.
- Fine-tuned: Qwen2.5-0.5B-Instruct fine-tuned using QLoRA.
- Task: Customer-support category and intent classification.
- Output requirement: JSON containing `category` and `intent`.

## Results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| JSON validity | {baseline['json_validity_rate'] * 100:.4f}% | {fine_tuned['json_validity_rate'] * 100:.4f}% |
| Schema validity | {baseline['schema_validity_rate'] * 100:.4f}% | {fine_tuned['schema_validity_rate'] * 100:.4f}% |
| Category label validity | {baseline['category_label_validity_rate'] * 100:.4f}% | {fine_tuned['category_label_validity_rate'] * 100:.4f}% |
| Intent label validity | {baseline['intent_label_validity_rate'] * 100:.4f}% | {fine_tuned['intent_label_validity_rate'] * 100:.4f}% |
| Category accuracy | {baseline['category_accuracy'] * 100:.4f}% | {fine_tuned['category_accuracy'] * 100:.4f}% |
| Intent accuracy | {baseline['intent_accuracy'] * 100:.4f}% | {fine_tuned['intent_accuracy'] * 100:.4f}% |
| Category macro F1 | {baseline['category_macro_f1'] * 100:.4f}% | {fine_tuned['category_macro_f1'] * 100:.4f}% |
| Intent macro F1 | {baseline['intent_macro_f1'] * 100:.4f}% | {fine_tuned['intent_macro_f1'] * 100:.4f}% |
| Category weighted F1 | {baseline['category_weighted_f1'] * 100:.4f}% | {fine_tuned['category_weighted_f1'] * 100:.4f}% |
| Intent weighted F1 | {baseline['intent_weighted_f1'] * 100:.4f}% | {fine_tuned['intent_weighted_f1'] * 100:.4f}% |
| Exact-match accuracy | {baseline['exact_match_accuracy'] * 100:.4f}% | {fine_tuned['exact_match_accuracy'] * 100:.4f}% |

## Interpretation

The baseline model produced valid JSON and satisfied the required output
schema, but frequently generated category and intent labels outside the
project's allowed label vocabulary.

QLoRA fine-tuning substantially improved the model's ability to produce
valid category and intent labels and match the expected classification
outputs on this test set.

The fine-tuned model achieved
**{fine_tuned['exact_match_accuracy'] * 100:.4f}% exact-match accuracy**,
compared with **{baseline['exact_match_accuracy'] * 100:.4f}%** for the
baseline.

## Evaluation considerations

These results describe performance on this project's held-out test set.
They should not automatically be interpreted as equivalent performance
on arbitrary real-world customer-support messages.

The evaluation split contains similar and paraphrased examples. Therefore,
the reported score should be considered in the context of the dataset
construction and test-set distribution.

## Generated artifacts

- `model_comparison.csv`: Complete metric comparison.
- `model_comparison.json`: Structured comparison data.
- `accuracy_comparison.png`: Accuracy comparison.
- `f1_comparison.png`: F1 comparison.
- `validity_comparison.png`: Output and label validity comparison.
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")


def main() -> None:
    """Generate all baseline-versus-fine-tuned comparison artifacts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_metrics(BASELINE_FILE)
    fine_tuned = load_metrics(FINE_TUNED_FILE)

    rows = build_comparison(baseline, fine_tuned)

    save_comparison_csv(rows)
    save_comparison_json(baseline, fine_tuned, rows)

    save_bar_chart(
        "accuracy_comparison.png",
        "Baseline vs Fine-tuned: Accuracy",
        [
            "category_accuracy",
            "intent_accuracy",
            "exact_match_accuracy",
        ],
        [
            "Category accuracy",
            "Intent accuracy",
            "Exact-match accuracy",
        ],
        baseline,
        fine_tuned,
    )

    save_bar_chart(
        "f1_comparison.png",
        "Baseline vs Fine-tuned: F1 Scores",
        [
            "category_macro_f1",
            "intent_macro_f1",
            "category_weighted_f1",
            "intent_weighted_f1",
        ],
        [
            "Category macro F1",
            "Intent macro F1",
            "Category weighted F1",
            "Intent weighted F1",
        ],
        baseline,
        fine_tuned,
    )

    save_bar_chart(
        "validity_comparison.png",
        "Baseline vs Fine-tuned: Output and Label Validity",
        [
            "json_validity_rate",
            "schema_validity_rate",
            "category_label_validity_rate",
            "intent_label_validity_rate",
        ],
        [
            "JSON validity",
            "Schema validity",
            "Category label validity",
            "Intent label validity",
        ],
        baseline,
        fine_tuned,
    )

    save_summary(baseline, fine_tuned)

    print("\nComparison artifacts generated successfully:")
    for path in [
        OUTPUT_CSV,
        OUTPUT_JSON,
        REPORT_DIR / "accuracy_comparison.png",
        REPORT_DIR / "f1_comparison.png",
        REPORT_DIR / "validity_comparison.png",
        OUTPUT_SUMMARY,
    ]:
        print(f" - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
