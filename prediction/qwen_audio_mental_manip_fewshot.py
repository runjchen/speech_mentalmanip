#!/usr/bin/env python3
import os
import re
import csv
import logging
import torch
import argparse
from collections import Counter
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

# -------------------
# QUIET NOISY LOGS
# -------------------
logging.getLogger().setLevel(logging.ERROR)

# -------------------
# CONFIG
# -------------------
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
AUDIO_DIR = "."                   # current directory
AUDIO_EXT = ".mp3"                # iterate only .mp3 files
OUTPUT_CSV = "0922_qwen_omni25_batch_results.csv"

MARGIN_YES_NO = 0.05              # margin for logits fallback (YES vs NO)
TACTIC_TIE_MARGIN = 0.03
MAX_EVIDENCE_WORDS = 12
N_VOTES = 5                       # YES/NO votes
N_TACTIC_VOTES = 5                # TACTIC votes

# ---- Few-shot calibration set (your exact paths + labels) ----
FEWSHOT = [
    ("./fewshot/85514440_composed_grandpa_grandma.mp3", "No"),
    ("./fewshot/85514449_composed_grandma_sassy.mp3", "No"),
    ("./fewshot/85514414_composed_amanda_mark.mp3", "Yes"),
    ("./fewshot/85514416_composed_ivanna_amanda.mp3", "Yes"),
]

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

VOCAB_SIZE = _get_vocab_size()
FULL_VOCAB = list(range(VOCAB_SIZE))

# -------------------
# PROMPTS
# -------------------
YESNO_PROMPT = (
    "You will first see FEW-SHOT examples with ground-truth labels. "
    "Use them only to calibrate your judgment, then answer for the TARGET.\n\n"
    "Task: Determine if the TARGET audio contains any form of mental manipulation.\n"
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
    "You saw FEW-SHOT examples above. Now classify the TARGET audio.\n"
    "Task: Choose the single best manipulation tactic for the TARGET from the list below.\n"
    "If there is no clear manipulation, choose 'none'.\n"
    "Options: " + ", ".join(TACTIC_LIST) + "\n"
    "Rule: Answer with exactly one option word from the list, nothing else.\n"
    "Answer:"
)

EVIDENCE_PROMPT_TMPL = (
    'You saw FEW-SHOT examples above. For the TARGET audio only, output ONE short quote '
    '(<=12 words) or paraphrase that supports the given tactic.\n'
    'Tactic: {tactic}\n'
    'CRITICAL RULES:\n'
    '1) Output ONLY the quote/paraphrase wrapped in double quotes.\n'
    '2) No prefixes like Reason:, Example:, Description:, Source:, Tactic:.\n'
    '3) No URLs, no extra text.\n'
    'Answer:'
)
EVIDENCE_PROMPT_RETRY_TMPL = (
    'Output a quote from the TARGET in double quotes. Nothing else.\n'
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

TACTIC_FIRST_IDS = {t: first_ids([t, " "+t, t.lower(), " "+t.lower()]) for t in TACTIC_LIST}
A_IDS = first_ids(["A", " A", "a", " a"])
B_IDS = first_ids(["B", " B", "b", " b"])
YES_IDS = first_ids(["YES","Yes","yes","YES.","Yes.","yes.","YES,","Yes,","yes,"])
NO_IDS  = first_ids(["NO","No","no","NO.","No.","no.","NO,","No,","no,"])

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
# CHAT TEMPLATE + FEWSHOT
# -------------------
DEFAULT_QWEN_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
def _abs(p): return os.path.abspath(p)

# Cache few-shot blocks so we don’t rebuild strings each call
_FEWSHOT_BLOCKS = None
def _fewshot_blocks():
    global _FEWSHOT_BLOCKS
    if _FEWSHOT_BLOCKS is not None:
        return _FEWSHOT_BLOCKS
    blocks = [{"type": "text", "text": "FEW-SHOT EXAMPLES (not the target).\nUse labels only for calibration."}]
    for path, lab in FEWSHOT:
        if lab != "No": continue
        if os.path.exists(path):
            blocks += [
                {"type": "text", "text": "\nExample (NOT manipulative):"},
                {"type": "audio", "path": _abs(path)},
                {"type": "text", "text": "Label: No"},
            ]
        else:
            blocks += [{"type": "text", "text": f"[WARN] Missing few-shot file: {path} (skipped)"}]
    for path, lab in FEWSHOT:
        if lab != "Yes": continue
        if os.path.exists(path):
            blocks += [
                {"type": "text", "text": "\nExample (manipulative):"},
                {"type": "audio", "path": _abs(path)},
                {"type": "text", "text": "Label: Yes"},
            ]
        else:
            blocks += [{"type": "text", "text": f"[WARN] Missing few-shot file: {path} (skipped)"}]
    blocks += [{"type": "text", "text": "\nNow analyze the TARGET audio below. Do NOT relabel examples above."}]
    _FEWSHOT_BLOCKS = blocks
    return blocks

def build_calibrated_inputs(target_audio_path: str, task_text: str):
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_QWEN_SYSTEM}]},
        {"role": "user",
         "content": _fewshot_blocks() + [
             {"type": "text", "text": "\nTARGET audio:"},
             {"type": "audio", "path": _abs(target_audio_path)},
             {"type": "text", "text": task_text},
         ]},
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
# YES/NO (single pass; no "uncertain")
# -------------------
def classify_yes_no_once(target_audio_path: str) -> str:
    inputs = build_calibrated_inputs(target_audio_path, YESNO_PROMPT)
    inp_len = inputs["input_ids"].size(-1)

    allowed_first = list(set(A_IDS + B_IDS))
    if allowed_first:
        def allow_ab_then_any(batch_id, input_ids):
            cur_len = input_ids.shape[-1]
            return allowed_first if cur_len == inp_len else FULL_VOCAB
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=3, do_sample=False, top_p=1.0,
                prefix_allowed_tokens_fn=allow_ab_then_any,
                return_dict_in_generate=True, output_scores=True,
            )
        seqs = gen.sequences
        new_ids = seqs[:, inp_len:] if seqs.dim() == 2 else seqs[inp_len:]
        text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        first = text.strip().splitlines()[0].strip().strip(".): ").upper()
        if first.startswith("A"): return "YES"
        if first.startswith("B"): return "NO"

    # Fallback: logits check on YES/NO
    prompt = "You saw FEW-SHOT examples. For the TARGET only, answer YES or NO.\nAnswer:"
    inputs = build_calibrated_inputs(target_audio_path, prompt)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=1, do_sample=False, top_p=1.0,
            return_dict_in_generate=True, output_scores=True,
        )
    logits = gen.scores[0]
    probs = torch.softmax(logits, dim=-1)
    vocab_size = probs.shape[-1]
    p_yes = float(sum(prob_at(probs, i) for i in YES_IDS if i < vocab_size)) if YES_IDS else 0.0
    p_no  = float(sum(prob_at(probs, i) for i in NO_IDS  if i < vocab_size)) if NO_IDS  else 0.0
    return "YES" if p_yes > (p_no + MARGIN_YES_NO) else "NO"

def classify_yes_no_majority(target_audio_path: str, n_votes: int = N_VOTES):
    votes = [classify_yes_no_once(target_audio_path) for _ in range(n_votes)]
    counts = Counter(votes)
    label = "YES" if counts["YES"] > counts["NO"] else "NO"
    return label, dict(counts)

# -------------------
# TACTIC (single pass + majority)
# -------------------
def classify_tactic_once(target_audio_path: str, forced_label: str):
    if forced_label != "YES":
        return "none"
    inputs = build_calibrated_inputs(target_audio_path, TACTIC_PROMPT)
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
        if not ids: continue
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

def classify_tactic_majority(target_audio_path: str, forced_label: str, n_votes: int = N_TACTIC_VOTES):
    if forced_label != "YES":
        return "none", {"none": n_votes}
    votes = [classify_tactic_once(target_audio_path, forced_label) for _ in range(n_votes)]
    counts = Counter(votes)
    # Conservative tie-break: prefer 'none' if tied
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0], dict(counts)
    top_count = most_common[0][1]
    tied = [t for t,c in most_common if c == top_count]
    if "none" in tied and len(tied) > 1:
        return "none", dict(counts)
    return most_common[0][0], dict(counts)

# -------------------
# EVIDENCE
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

def gen_evidence(target_audio_path: str, tactic: str) -> str:
    if tactic == "none":
        return ""
    def ask(prompt):
        inputs = build_calibrated_inputs(target_audio_path, prompt)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=24, do_sample=False, top_p=1.0,
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
# MAIN (iterate all .mp3; write CSV once at the end)
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Few-shot calibrate with Qwen, then batch-analyze all .mp3 files (majority-vote YES/NO & TACTIC).")
    parser.add_argument("--votes", type=int, default=N_VOTES, help=f"YES/NO runs for majority vote (default: {N_VOTES})")
    parser.add_argument("--tactic-votes", type=int, default=N_TACTIC_VOTES, help=f"TACTIC runs for majority vote (default: {N_TACTIC_VOTES})")
    parser.add_argument("--out", default=OUTPUT_CSV, help=f"Output CSV filename (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    mp3s = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(AUDIO_EXT)])
    if not mp3s:
        print(f"No {AUDIO_EXT} files found in {AUDIO_DIR}")
        return

    rows = []
    print(f"Found {len(mp3s)} {AUDIO_EXT} files. Starting batch…\n")

    for fname in mp3s:
        fpath = os.path.join(AUDIO_DIR, fname)
        try:
            # YES/NO majority
            label, counts_yesno = classify_yes_no_majority(fpath, n_votes=args.votes)
            # TACTIC majority (only if YES)
            tactic, counts_tactic = classify_tactic_majority(fpath, label, n_votes=args.tactic_votes)
            tech_type = to_technique_type(tactic)
            evidence = ""
            if label == "YES" and tactic != "none":
                try:
                    evidence = gen_evidence(fpath, tactic)
                except Exception as e:
                    print(f"[EVIDENCE ERROR] {fname}: {e}")
                    evidence = "reason_failed"

            uses_any = "Yes" if (label == "YES" and tactic != "none") else "No"

            # Console strict 3-line per file
            print(f"=== {fname} ===")
            print("Yes" if label == "YES" else "No")
            print(tactic)
            print(f"\"{evidence}\"" if evidence else "")
            print(f"YES/NO votes: {counts_yesno}")
            if label == "YES":
                print(f"TACTIC votes: {counts_tactic}")
            print()

            rows.append([fname, label, tactic, tech_type, evidence, uses_any])

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            rows.append([fname, "ERROR", "none", "none", str(e), "No"])

    # Write CSV once at the end
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","answer","tactic","technique_type","evidence","uses_any"])
        w.writerows(rows)

    print(f"Saved results to {args.out}")

if __name__ == "__main__":
    main()

