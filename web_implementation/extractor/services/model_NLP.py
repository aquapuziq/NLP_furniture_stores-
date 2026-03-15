from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = Path(r"C:\dev\projects\NLP_test\res\final_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast = True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def binary_tags_to_spans(offsets, pred_ids, id2label):
    spans = []
    current_start = None
    current_end = None

    for (start, end), pred_id in zip(offsets, pred_ids):
        if start == 0 and end == 0:
            continue

        tag = id2label[pred_id]

        if tag == "PRODUCT":
            if current_start is None:
                current_start = start
                current_end = end
            else:
                current_end = end
        else:
            if current_start is not None:
                spans.append((current_start, current_end))
                current_start = None
                current_end = None

    if current_start is not None:
        spans.append((current_start, current_end))

    return spans


def merge_overlapping_spans(spans):
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [spans[0]]

    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def normalize_product_text(text: str) -> str:
    return " ".join(text.split()).strip(" -–—,:;|")


def predict_products_from_text(text, max_length = 256, stride = 64):
    enc = tokenizer(text, truncation = True, padding = "max_length", max_length = max_length, stride = stride,
        return_overflowing_tokens = True, return_offsets_mapping = True, return_tensors = "pt")

    offset_mapping = enc["offset_mapping"].tolist()
    model_inputs = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
    id2label = model.config.id2label

    all_spans = []
    for offsets, pred_ids in zip(offset_mapping, predictions):
        spans = binary_tags_to_spans(offsets, pred_ids, id2label)
        all_spans.extend(spans)

    all_spans = merge_overlapping_spans(all_spans)

    products = []
    seen = set()

    for start, end in all_spans:
        value = normalize_product_text(text[start:end])
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            products.append(value)

    return products