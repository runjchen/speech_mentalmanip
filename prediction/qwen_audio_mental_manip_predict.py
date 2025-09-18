#!/usr/bin/env python3
import os
import re
import csv
import logging
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

# -------------------
# QUIET NOISY LOGS
# -------------------
logging.getLogger().setLevel(logging.ERROR)

# -------------------
# CONFIG
# -------------------
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
AUDIO_DIR = "."
OUTPUT_CSV = "qwen_omni25_results_with_types.csv"

MARGIN_YES_NO = 0.05        # extra margin to say YES over NO
UNCERTAIN_MARGIN = 0.08     # if |p_yes - p_no| < this -> UNCERTAIN
TACTIC_TIE_MARGIN = 0.03
MAX_EVIDENCE_WORDS = 12
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

# -------------------
# LOAD MODEL
# -------------------
print("Loading Qwen2.5-Omni (Thinker-only)...")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
).eval()
try:
    model.tie_weights()
except Exception:
    pass

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

# Robust vocab size: prefer embedding matrix, else text_config, else tokenizer
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
    return 32000  # last-resort default

VOCAB_SIZE = _get_vocab_size()
FULL_VOCAB = list(range(VOCAB_SIZE))

# -------------------
# PROMPTS
# -------------------
YESNO_PROMPT = (
    "Task: Determine if the audio contains any form of mental manipulation.\n"
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
    "Task: Choose the single best manipulation tactic for this audio from the list below.\n"
    "If there is no clear manipulation, choose 'none'.\n"
    "Options: " + ", ".join(TACTIC_LIST) + "\n"
    "Rule: Answer with exactly one option word from the list, nothing else.\n"
    "Answer:"
)

EVIDENCE_PROMPT_TMPL = (
    'Task: Output ONE short quote or paraphrase (<=12 words) from this audio '
    'that supports the given tactic.\n'
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

# -------------------
# HELPERS
# -------------------
def uniq_preserve(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def first_ids(variants):
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

# token id sets for tactics (logit trick works well here)
TACTIC_FIRST_IDS = {t: first_ids([t, " "+t, t.lower(), " "+t.lower()]) for t in TACTIC_LIST}

# A/B token id sets (for constrained first token + calibrated probs)
A_IDS = first_ids(["A", " A", "a", " a"])
B_IDS = first_ids(["B", " B", "b", " b"])

# fallback YES/NO ids if we ever need to peek single token
YES_IDS = first_ids(["YES","Yes","yes","YES.","Yes.","yes.","YES,","Yes,","yes,"])
NO_IDS  = first_ids(["NO","No","no","NO.","No.","no.","NO,","No,","no,"])

# -------------------
# MAPPINGS
# -------------------
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
# CHAT TEMPLATE
# -------------------
DEFAULT_QWEN_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

def build_inputs(audio_path: str, text_prompt: str):
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_QWEN_SYSTEM}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": os.path.abspath(audio_path)},
                {"type": "text", "text": text_prompt},
            ],
        },
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
# YES/NO via DISCRETE A/B (constrain 1st token; full vocab thereafter)
# -------------------
def classify_yes_no(audio_path: str):
    inputs = build_inputs(audio_path, YESNO_PROMPT)
    inp_len = inputs["input_ids"].size(-1)

    allowed_first = list(set(A_IDS + B_IDS))
    if not allowed_first:
        return classify_yes_no_logits(audio_path)

    def allow_ab_then_any(batch_id, input_ids):
        # input_ids may be (cur_len,) or (1, cur_len)
        cur_len = input_ids.shape[-1]
        # First generated token -> A/B only; after that -> full vocab
        return allowed_first if cur_len == inp_len else FULL_VOCAB

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=3,            # enough to emit "A" or "B"
            do_sample=False,
            top_p=1.0,
            prefix_allowed_tokens_fn=allow_ab_then_any,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Calibrated p(A)/p(B) from first step
    if not getattr(gen, "scores", None):
        return classify_yes_no_logits(audio_path)

    logits = gen.scores[0]               # [vocab] or [1, vocab]
    probs  = torch.softmax(logits, dim=-1)
    vocab_size = probs.shape[-1]

    p_a = sum(prob_at(probs, i) for i in A_IDS if i < vocab_size) if A_IDS else 0.0
    p_b = sum(prob_at(probs, i) for i in B_IDS if i < vocab_size) if B_IDS else 0.0

    # Decode written answer (should start with "A" or "B")
    seqs = gen.sequences
    new_ids = seqs[:, inp_len:] if seqs.dim() == 2 else seqs[inp_len:]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
    first = text.strip().splitlines()[0].strip().strip(".): ").upper()

    if first.startswith("A"):
        p_yes, p_no = p_a, p_b
    elif first.startswith("B"):
        p_yes, p_no = p_a, p_b
    else:
        return classify_yes_no_logits(audio_path)

    if abs(p_yes - p_no) < UNCERTAIN_MARGIN:
        label = "UNCERTAIN"
    else:
        label = "YES" if p_yes > (p_no + MARGIN_YES_NO) else "NO"
    return label, float(p_yes), float(p_no)

def classify_yes_no_logits(audio_path: str):
    """Fallback: direct YES/NO single-token check."""
    prompt = "Answer YES or NO only.\nAnswer:"
    inputs = build_inputs(audio_path, prompt)
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
    p_yes = float(sum(prob_at(probs, i) for i in YES_IDS if i < vocab_size)) if YES_IDS else 0.0
    p_no  = float(sum(prob_at(probs, i) for i in NO_IDS  if i < vocab_size)) if NO_IDS  else 0.0
    if abs(p_yes - p_no) < UNCERTAIN_MARGIN:
        return "UNCERTAIN", p_yes, p_no
    label = "YES" if p_yes > (p_no + MARGIN_YES_NO) else "NO"
    return label, p_yes, p_no

# -------------------
# TACTIC CLASSIFIER (one-word options = stable with first-token logit)
# -------------------
def classify_tactic(audio_path: str, forced_label: str):
    if forced_label != "YES":
        return "none"
    inputs = build_inputs(audio_path, TACTIC_PROMPT)
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

    scores = []
    for tactic, ids in TACTIC_FIRST_IDS.items():
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

# -------------------
# EVIDENCE GENERATION
# -------------------
def _clean_evidence_text(txt: str) -> str:
    t = txt.replace("\n", " ").strip()
    m = re.search(r'"([^"]+)"', t)
    if m:
        t = m.group(1)
    bad = ("reason:", "example:", "description:", "tactic:", "source:", "effect:", "audio:")
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

def gen_evidence(audio_path: str, tactic: str) -> str:
    if tactic == "none":
        return ""
    def ask(prompt):
        inputs = build_inputs(audio_path, prompt)
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
    ev = _clean_evidence_text(text)
    if not ev:
        text = ask(EVIDENCE_PROMPT_RETRY_TMPL.format(tactic=tactic))
        ev = _clean_evidence_text(text)
    return ev

# -------------------
# MAIN
# -------------------
def main():
    rows = []
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if not fname.lower().endswith(AUDIO_EXTS):
            continue
        fpath = os.path.join(AUDIO_DIR, fname)
        try:
            label, p_yes, p_no = classify_yes_no(fpath)
        except Exception as e:
            print(f"[CLASSIFY ERROR] {fname}: {e}")
            rows.append([fname, "ERROR", "0.0000", "0.0000", "none", "none", str(e), "No"])
            continue

        tactic = classify_tactic(fpath, label)
        tech_type = to_technique_type(tactic)

        try:
            evidence = gen_evidence(fpath, tactic) if label == "YES" else ""
        except Exception as e:
            print(f"[EVIDENCE ERROR] {fname}: {e}")
            evidence = "reason_failed"

        uses_any = "Yes" if (label == "YES" and tactic != "none") else "No"
        rows.append([
            fname,
            label,
            f"{p_yes:.4f}",
            f"{p_no:.4f}",
            tactic,
            tech_type,
            evidence,
            uses_any
        ])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","answer","p_yes","p_no","tactic","technique_type","evidence","uses_any"])
        w.writerows(rows)

    print(f"Saved {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

