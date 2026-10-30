# SupportSFT: Baseline vs Fine-tuned Model

## Evaluation setup

Both models were evaluated on the same **2,428 test examples**.

- Baseline: Original Qwen2.5-0.5B-Instruct model.
- Fine-tuned: Qwen2.5-0.5B-Instruct fine-tuned using QLoRA.
- Task: Customer-support category and intent classification.
- Output requirement: JSON containing `category` and `intent`.

## Results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| JSON validity | 100.0000% | 100.0000% |
| Schema validity | 100.0000% | 100.0000% |
| Category label validity | 8.0725% | 100.0000% |
| Intent label validity | 0.0824% | 100.0000% |
| Category accuracy | 6.1779% | 99.9588% |
| Intent accuracy | 0.0824% | 99.9588% |
| Category macro F1 | 11.5452% | 99.9573% |
| Intent macro F1 | 0.1623% | 99.9576% |
| Category weighted F1 | 10.2889% | 99.9588% |
| Intent weighted F1 | 0.1629% | 99.9588% |
| Exact-match accuracy | 0.0412% | 99.9588% |

## Interpretation

The baseline model produced valid JSON and satisfied the required output
schema, but frequently generated category and intent labels outside the
project's allowed label vocabulary.

QLoRA fine-tuning substantially improved the model's ability to produce
valid category and intent labels and match the expected classification
outputs on this test set.

The fine-tuned model achieved
**99.9588% exact-match accuracy**,
compared with **0.0412%** for the
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
