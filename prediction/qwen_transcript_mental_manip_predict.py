#!/usr/bin/env python3
"""
Text-only manipulation detection from Dialogue in mental_con.csv,
restricted to files present in a local directory.

Output columns (exactly):
filename,answer,p_yes,p_no,tactic,technique_type,evidence,uses_any

Usage:
  python *.py \
    --csv mental_con.csv \
    --out qwen_omni25_results_with_types.csv \
    --dir . \
    [--model Qwen/Qwen2.5-Omni-7B]
"""

import os
import re
import csv
import argparse
import logging
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

# -------------------
# QUIET NOISY LOGS
# -------------------
logging.getLogger().setLevel(logging.ERROR)

# -------------------
# CONFIG / CONSTANTS
# -------------------
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
MARGIN_YES_NO = 0.05        # extra margin to say YES over NO
UNCERTAIN_MARGIN = 0.08     # if |p_yes - p_no| < this -> UNCERTAIN
TACTIC_TIE_MARGIN = 0.03
MAX_EVIDENCE_WORDS = 12

DEFAULT_QWEN_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

YESNO_PROMPT = (
    "Task: Determine if the Dialogue contains any form of mental manipulation.\n"
    "Choose exactly one option and output ONLY the letter on the first line.\n"
    "(A) YES — clear evidence of manipulation\n"
    "(B) NO  — otherwise\n"
    "Answer (A or B):"
)

TACTIC_LIST = [
    "Accusation","Brandishing Anger","Denial","Evasion","Feigning Innocence",
    "Intimidation","Persuasion or Seduction","Playing Servant Role",
    "Playing Victim Role","Rationalization","Shaming or Belittlement","none"
]

TACTIC_PROMPT = (
    "Task: Choose the single best manipulation tactic for this Dialogue from the list below.\n"
    "If there is no clear manipulation, choose 'none'.\n"
    "Options: " + ", ".join(TACTIC_LIST) + "\n"
    "Rule: Answer with exactly one option word from the list, nothing else.\n"
    "Answer:"
)

EVIDENCE_PROMPT_TMPL = (
    'Task: From the Dialogue above, output ONE short quote or paraphrase (<=12 words) '
    'that best supports the given tactic.\n'
    'Tactic: {tactic}\n'
    'CRITICAL RULES:\n'
    '1) Output ONLY the quote/paraphrase wrapped in double quotes.\n'
    '2) No prefixes like Reason:, Example:, Description:, Source:, Tactic:.\n'
    '3) No URLs, no extra text.\n'
    'Answer:'
)
EVIDENCE_PROMPT_RETRY_TMPL = (
    'Output ONLY a short quote (<=12 words) in double quotes. Nothing else.\n'
    'Tactic: {tactic}\n'
    'Answer:'
)

LIKELY_FILENAME_COLS = ["filename", "file", "file_name", "fname", "audio", "path", "name"]
LIKELY_TEXT_COLS     = ["dialogue", "transcript", "text", "content", "utterance", "utterances", "lines", "script", "conversation", "dialog"]

TECHNIQUE_TYPE_MAP = {
    "Accusation": "Blame/Shame",
    "Shaming or Belittlement": "Blame/Shame",
    "Brandishing Anger": "Coercion/Threat",
    "Intimidation": "Coercion/Threat",
    "Denial": "Deception/Evasion",
    "Evasion": "Deception/Evasion",
    "Feigning Innocence": "Deception/Evasion",
    "Rationalization": "Justification/Minimization",
    "Persuasion or Seduction": "Charm/Reward",
    "Playing Servant Role": "Self-Presentation",
    "Playing Victim Role": "Self-Presentation",
    "none": "none",
}

def to_technique_type(tactic: str) -> str:
    return TECHNIQUE_TYPE_MAP.get(tactic, "Other")

# -------------------
# SMALL HELPERS
# -------------------
def find_col(header, candidates):
    lower = {h.lower(): h for h in header}
    for c in candidates:
        if c in lower:
            return lower[c]
    # fuzzy match
    for h in header:
        hl = h.lower()
        for c in candidates:
            if c in hl:
                return h
    return None

def id_from_filename(fname: str) -> str:
    m = re.match(r"^\s*(\d+)", os.path.basename(fname))
    return m.group(1) if m else os.path.basename(fname)

def uniq_preserve(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def first_ids(tokenizer, variants):
    ids = []
    for s in variants:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return uniq_preserve(ids)

def prob_at(probs: torch.Tensor, idx: int) -> float:
    if probs.dim() == 2:
        return float(probs[0, idx])
    return float(probs[idx])

def clean_evidence_text(txt: str) -> str:
    t = txt.replace("\n", " ").strip()
    m = re.search(r'"([^"]+)"', t)
    if m:
        t = m.group(1)
    bad = ("reason:", "example:", "description:", "tactic:", "source:", "effect:", "audio:", "quote:")
    tl = t.lower().lstrip()
    for p in bad:
        if tl.startswith(p):
            t = t[len(p):].lstrip(" :,-")
            break
    t = re.sub(r'https?://\S+', '', t).strip()
    words = t.split()
    if len(words) > MAX_EVIDENCE_WORDS:
        t = " ".join(words[:MAX_EVIDENCE_WORDS]) + "…"
    return t.strip(' "\'')

# -------------------
# MODEL SETUP
# -------------------
def load_model(model_name: str):
    print("Loading Qwen2.5-Omni (Thinker-only) for TEXT...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    ).eval()
    try:
        model.tie_weights()
    except Exception:
        pass
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # Robust vocab size
    def _get_vocab_size():
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return int(emb.weight.shape[0])
        except Exception:
            pass
        try:
            return int(getattr(getattr(model.config, "text_config", model.config), "vocab_size"))
        except Exception:
            pass
        try:
            return int(tokenizer.vocab_size)
        except Exception:
            pass
        return 32000
    vocab_size = _get_vocab_size()
    full_vocab = list(range(vocab_size))
    return model, processor, tokenizer, full_vocab

def build_text_inputs(processor, model, dialogue_text: str, text_prompt: str):
    content = f'Dialogue:\n"""\n{dialogue_text.strip()}\n"""\n\n{text_prompt}'
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_QWEN_SYSTEM}]},
        {"role": "user",   "content": [{"type": "text", "text": content}]},
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    return inputs

# -------------------
# DISCRETE A/B — YES/NO
# -------------------
def make_yesno_ids(tokenizer):
    a_ids = first_ids(tokenizer, ["A", " A", "a", " a"])
    b_ids = first_ids(tokenizer, ["B", " B", "b", " b"])
    yes_ids = first_ids(tokenizer, ["YES","Yes","yes","YES.","Yes.","yes.","YES,","Yes,","yes,"])
    no_ids  = first_ids(tokenizer, ["NO","No","no","NO.","No.","no.","NO,","No,","no,"])
    return a_ids, b_ids, yes_ids, no_ids

def classify_dialogue_yesno(model, processor, tokenizer, full_vocab, dialogue_text: str):
    a_ids, b_ids, yes_ids, no_ids = make_yesno_ids(tokenizer)
    inputs = build_text_inputs(processor, model, dialogue_text, YESNO_PROMPT)
    inp_len = inputs["input_ids"].size(-1)

    allowed_first = list(set(a_ids + b_ids))
    if not allowed_first:
        return classify_dialogue_yesno_fallback(model, processor, tokenizer, dialogue_text, yes_ids, no_ids)

    def allow_ab_then_any(batch_id, input_ids):
        cur_len = input_ids.shape[-1]
        return allowed_first if cur_len == inp_len else full_vocab

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            top_p=1.0,
            prefix_allowed_tokens_fn=allow_ab_then_any,
            return_dict_in_generate=True,
            output_scores=True,
        )

    if not getattr(gen, "scores", None):
        return classify_dialogue_yesno_fallback(model, processor, tokenizer, dialogue_text, yes_ids, no_ids)

    logits = gen.scores[0]
    probs  = torch.softmax(logits, dim=-1)
    vocab_size = probs.shape[-1]

    p_a = sum(prob_at(probs, i) for i in a_ids if i < vocab_size) if a_ids else 0.0
    p_b = sum(prob_at(probs, i) for i in b_ids if i < vocab_size) if b_ids else 0.0

    seqs = gen.sequences
    new_ids = seqs[:, inp_len:] if seqs.dim() == 2 else seqs[inp_len:]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
    first = text.strip().splitlines()[0].strip().strip(".): ").upper()

    if first.startswith("A"):
        p_yes, p_no = p_a, p_b
    elif first.startswith("B"):
        p_yes, p_no = p_a, p_b
    else:
        return classify_dialogue_yesno_fallback(model, processor, tokenizer, dialogue_text, yes_ids, no_ids)

    if abs(p_yes - p_no) < UNCERTAIN_MARGIN:
        label = "UNCERTAIN"
    else:
        label = "YES" if p_yes > (p_no + MARGIN_YES_NO) else "NO"
    return label, float(p_yes), float(p_no)

def classify_dialogue_yesno_fallback(model, processor, tokenizer, dialogue_text: str, yes_ids, no_ids):
    prompt = 'Answer YES or NO only.\nAnswer:'
    inputs = build_text_inputs(processor, model, dialogue_text, prompt)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if not hasattr(gen, "scores") or not gen.scores:
        return "UNCERTAIN", 0.5, 0.5
    logits = gen.scores[0]
    probs = torch.softmax(logits, dim=-1)
    vocab_size = probs.shape[-1]
    p_yes = float(sum(prob_at(probs, i) for i in yes_ids if i < vocab_size)) if yes_ids else 0.0
    p_no  = float(sum(prob_at(probs, i) for i in  no_ids if i < vocab_size)) if  no_ids else 0.0
    if abs(p_yes - p_no) < UNCERTAIN_MARGIN:
        return "UNCERTAIN", p_yes, p_no
    return ("YES" if p_yes > (p_no + MARGIN_YES_NO) else "NO"), p_yes, p_no

# -------------------
# TACTIC + EVIDENCE (text-only)
# -------------------
def classify_tactic_text(model, processor, tokenizer, dialogue_text: str):
    inputs = build_text_inputs(processor, model, dialogue_text, TACTIC_PROMPT)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if not hasattr(gen, "scores") or not gen.scores:
        return "none"
    logits = gen.scores[0]
    probs = torch.softmax(logits, dim=-1)
    vocab_size = probs.shape[-1]

    # token id sets for tactic first tokens
    tactic_first_ids = {t: first_ids(tokenizer, [t, " "+t, t.lower(), " "+t.lower()]) for t in TACTIC_LIST}
    scores = []
    for tactic, ids in tactic_first_ids.items():
        if not ids:
            continue
        p = float(sum(prob_at(probs, i) for i in ids if i < vocab_size))
        scores.append((p, tactic))
    if not scores:
        return "none"
    scores.sort(reverse=True)
    best_p, best_t = scores[0]
    if len(scores) > 1:
        second_p, second_t = scores[1]
        if best_t == "none" or (best_p - second_p) < TACTIC_TIE_MARGIN:
            best_t = second_t
    return best_t

def gen_evidence_text(model, processor, tokenizer, dialogue_text: str, tactic: str) -> str:
    if tactic == "none":
        return ""
    def ask(prompt):
        inputs = build_text_inputs(processor, model, dialogue_text, prompt)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                top_p=1.0,
            )
        new_ids = out_ids[:, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_ids[0], skip_special_tokens=True)

    text = ask(EVIDENCE_PROMPT_TMPL.format(tactic=tactic))
    ev = clean_evidence_text(text)
    if not ev:
        text = ask(EVIDENCE_PROMPT_RETRY_TMPL.format(tactic=tactic))
        ev = clean_evidence_text(text)
    return ev

# -------------------
# MAIN
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="mental_con.csv", help="Path to mental_con.csv")
    ap.add_argument("--out",  default="qwen_omni25_results_with_types.csv", help="Output CSV path")
    ap.add_argument("--dir",  default=".", help="Directory containing local audio files to filter by")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B", help="HF model id")
    args = ap.parse_args()

    # Build set/dicts of files present in local dir
    present_files = set()
    id_to_filename = {}
    for fn in os.listdir(args.dir):
        if fn.lower().endswith(AUDIO_EXTS):
            present_files.add(fn)
            _id = id_from_filename(fn)
            id_to_filename.setdefault(_id, fn)  # keep first seen

    if not present_files:
        raise RuntimeError(f"No audio files with {AUDIO_EXTS} found in {args.dir}")

    # Open CSV, find columns
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"{args.csv} not found")

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise RuntimeError("CSV has no header/columns")

        fname_col = find_col(header, [c.lower() for c in LIKELY_FILENAME_COLS])
        text_col  = find_col(header, [c.lower() for c in LIKELY_TEXT_COLS])

        if not fname_col and "id" in [h.lower() for h in header]:
            # allow pure id matching via 'id'
            fname_col = [h for h in header if h.lower() == "id"][0]

        if not fname_col:
            raise RuntimeError(f"Could not find a filename column among: {LIKELY_FILENAME_COLS} (found: {header})")
        if not text_col:
            raise RuntimeError(f"Could not find a dialogue text column among: {LIKELY_TEXT_COLS} (found: {header})")

        # Load model once
        model, processor, tokenizer, full_vocab = load_model(args.model)

        rows_out = []
        n_total = 0
        n_considered = 0
        n_missing_in_dir = 0
        n_skipped_blank = 0

        for row in reader:
            n_total += 1
            raw_name = (row.get(fname_col) or "").strip()
            text     = (row.get(text_col) or "").strip()
            if not raw_name or not text:
                n_skipped_blank += 1
                continue

            # Normalize: try exact filename first
            base = os.path.basename(raw_name)
            candidate = base if base in present_files else None

            # If not exact, try by leading numeric ID
            if candidate is None:
                _id = id_from_filename(base)
                candidate = id_to_filename.get(_id)

            if candidate is None or candidate not in present_files:
                n_missing_in_dir += 1
                continue  # skip rows whose file isn't in local dir

            n_considered += 1
            filename = candidate

            try:
                label, p_yes, p_no = classify_dialogue_yesno(model, processor, tokenizer, full_vocab, text)
            except Exception as e:
                print(f"[CLASSIFY ERROR] {filename}: {e}")
                rows_out.append([filename, "ERROR", "0.0000", "0.0000", "none", "none", str(e), "No"])
                continue

            tactic = classify_tactic_text(model, processor, tokenizer, text) if label == "YES" else "none"
            tech_type = to_technique_type(tactic)
            try:
                evidence = gen_evidence_text(model, processor, tokenizer, text, tactic) if label == "YES" else ""
            except Exception as e:
                print(f"[EVIDENCE ERROR] {filename}: {e}")
                evidence = "reason_failed"

            uses_any = "Yes" if (label == "YES" and tactic != "none") else "No"
            rows_out.append([
                filename,
                label,
                f"{p_yes:.4f}",
                f"{p_no:.4f}",
                tactic,
                tech_type,
                evidence,
                uses_any
            ])

    # Write results with the EXACT header you expect
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","answer","p_yes","p_no","tactic","technique_type","evidence","uses_any"])
        w.writerows(rows_out)

    print(f"Saved {args.out}")
    print(f"Rows in CSV: {n_total} | considered (present in dir): {n_considered} | "
          f"missing_in_dir: {n_missing_in_dir} | skipped_blank: {n_skipped_blank}")


if __name__ == "__main__":
    main()
