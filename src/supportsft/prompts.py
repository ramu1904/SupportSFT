
import json

SYSTEM_PROMPT = (
    "Classify customer support requests. "
    'Return only a JSON object with keys "category" and "intent".'
)

REQUIRED_LABEL_COLUMNS = {"instruction", "category", "intent"}


def _validate_text(value, field_name: str) -> str:
    """Convert a value to clean text and reject empty or missing values."""
    if value is None:
        raise ValueError(f"{field_name} must not be empty.")

    text = str(value).strip()

    if not text or text.casefold() in {"nan", "none"}:
        raise ValueError(f"{field_name} must not be empty.")

    return text


def build_inference_messages(instruction: str) -> list[dict[str, str]]:
    """Build system and user messages for an inference request."""
    instruction = _validate_text(instruction, "Instruction")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def build_training_messages(row) -> list[dict[str, str]]:
    """Build system, user, and assistant messages for a labeled example."""
    missing_columns = REQUIRED_LABEL_COLUMNS - set(row.index)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    instruction = _validate_text(row["instruction"], "Instruction")
    category = _validate_text(row["category"], "Category")
    intent = _validate_text(row["intent"], "Intent")

    target = json.dumps(
        {"category": category, "intent": intent},
        ensure_ascii=False,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": target},
    ]


def format_training_conversation(row, tokenizer) -> str:
    """Format a complete labeled conversation with the tokenizer's template."""
    messages = build_training_messages(row)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def format_inference_prompt(instruction: str, tokenizer) -> str:
    """Format an inference prompt and add the assistant-generation prefix."""
    messages = build_inference_messages(instruction)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
