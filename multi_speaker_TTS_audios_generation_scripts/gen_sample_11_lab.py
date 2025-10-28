#!/usr/bin/env python3
"""
Audio generation pipeline using the ElevenLabs SDK and a CSV for conversation data.

- Processes rows where Manipulative matches the provided label (0 or 1).
- For each conversation:
    * Randomly pick exactly two voices from VOICE_SLOTS.
    * Assign one to Person1, the other to Person2.
    * Generate each turn with its mapped voice.
- Skips files already in ./samples/final/
"""

import sys, os, argparse, pandas as pd, re, random
from pathlib import Path
from elevenlabs.client import ElevenLabs

VOICE_SLOTS = [
    {"slug": "ivanna",  "name": "Ivanna - Young & Casual",     "voice_id": "yM93hbw8Qtvdma2wCnJG"},
    {"slug": "mark",    "name": "Mark - Natural Conversations", "voice_id": "UgBBYS2sOqTuMpoF3BR0"},
    {"slug": "amanda",  "name": "Amanda",                       "voice_id": "M6N6IdXhi5YNZyZSDe7k"},
    {"slug": "grandpa", "name": "Grandpa Spuds Oxley",          "voice_id": "NOpBlnGInO9m6vDvFkFC"},
    {"slug": "grandma", "name": "Grandma Muffin",               "voice_id": "vFLqXa8bgbofGarf6fZh"},
    {"slug": "sassy",   "name": "Sassy Aerisita",               "voice_id": "03vEurziQfq3V8WZhQvn"},
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-speaker TTS (two voices) from CSV."
    )
    p.add_argument("-CONV", "--conversation-id", required=True, help="Conversation ID")
    p.add_argument("-CSV",  "--csv",             required=True, help="Path to mentalmanip_con.csv")
    p.add_argument("-o",    "--output",          default="./samples/", help="Output dir")
    p.add_argument("-k",    "--api-key",         help="ElevenLabs API key (or set env)")
    p.add_argument("-M",    "--manipulative",    type=int, default=1, help="Manipulative label (0 or 1)")
    return p.parse_args()

def fetch_turns(csv_file, conv_id, manip_label):
    df = pd.read_csv(csv_file)
    df = df[(df["Manipulative"] == manip_label) & (df["ID"].astype(str) == str(conv_id))]
    if df.empty:
        raise ValueError(f"No dialogue with Manipulative=={manip_label} for {conv_id}")
    text = df.iloc[0]["Dialogue"]
    pattern = re.compile(r'(Person\d+):\s*([^\n]+)')
    turns = [{"speaker": m.group(1), "text": m.group(2).strip(), "turn": i}
             for i, m in enumerate(pattern.finditer(text), start=1)]
    if not turns:
        raise ValueError(f"No turns parsed for {conv_id}")
    return turns

def generate(client, conv_id, csv_file, out_root, manip_label):
    turns = fetch_turns(csv_file, conv_id, manip_label)

    # extract speaker labels (should be ["Person1","Person2"])
    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) != 2:
        raise ValueError(f"Expected exactly 2 speakers, got {speakers}")

    # pick two distinct voices
    slots = random.sample(VOICE_SLOTS, 2)
    # map Person1->slots[0], Person2->slots[1]
    mapping = { speakers[i]: slots[i] for i in (0,1) }
    print(f"🎛  {conv_id}: speaker→voice = {{ {speakers[0]}: {slots[0]['slug']}, {speakers[1]}: {slots[1]['slug']} }}")

    out_dir   = Path(out_root)
    final_dir = out_dir / "final"
    out_dir.mkdir(exist_ok=True, parents=True)
    final_dir.mkdir(exist_ok=True, parents=True)

    for t in turns:
        slot   = mapping[t["speaker"]]
        fname  = f"{slot['slug']}_{conv_id}_{t['turn']}.mp3"
        out_mp3 = out_dir / fname
        skip_mp3= final_dir / fname

        if skip_mp3.exists():
            print("⏭️ Skipping existing:", fname)
            continue

        print("🎙️ Generating:", fname, "with", slot["name"])
        ssml = f"<speak>{t['text']}<break time=\"1s\"/></speak>"
        stream = client.text_to_speech.convert(
            text=ssml,
            voice_id=slot["voice_id"],
            model_id="eleven_multilingual_v2",
            voice_settings={"ssml":True},
            output_format="mp3_44100_128"
        )
        with open(out_mp3, "wb") as f:
            for chunk in stream:
                f.write(chunk)

    print("✅ Done", conv_id)

def main():
    args = parse_args()
    key = args.api_key or os.getenv("ELEVEN_LABS_API_KEY")
    if not key:
        sys.exit("ERROR: missing API key")
    client = ElevenLabs(api_key=key)
    try:
        generate(client, args.conversation_id, args.csv, args.output, args.manipulative)
    except Exception as e:
        sys.exit(f"Generation failed: {e}")

if __name__=="__main__":
    main()

