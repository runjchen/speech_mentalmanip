#!/usr/bin/env python3
"""
Round-Robin Composer for 2-voice conversations with ordered slugs.

Scans <input_dir> for files named:
    <voice_slug>_<conversationID>_<turn>.mp3

Keeps only conversations where exactly two slugs appear,
determines slug1 from the speaker on turn 1 and slug2 from speaker on turn 2,
loads each turn in ascending order (skipping unreadable files),
concatenates with 500 ms silence between turns,
and writes:
    <conversationID>_composed_<slug1>_<slug2>.mp3
"""

import argparse
import os
import re
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

FILE_RE = re.compile(r'^(?P<slug>[a-z]+)_(?P<conv>\d+)_(?P<turn>\d+)\.mp3$')

def compose_one(conv_id: str, inp: str, outp: str):
    # Gather all matching files
    items = []
    for fn in os.listdir(inp):
        m = FILE_RE.match(fn)
        if not m or m.group('conv') != conv_id:
            continue
        turn = int(m.group('turn'))
        slug = m.group('slug')
        items.append((turn, slug, fn))

    if not items:
        print(f"⚠️  No files for conversation {conv_id}, skipping.")
        return

    # Sort by turn
    items.sort(key=lambda x: x[0])

    # Determine slugs by first appearance order
    seen = []
    for _, slug, _ in items:
        if slug not in seen:
            seen.append(slug)
        if len(seen) == 2:
            break
    if len(seen) != 2:
        print(f"⚠️  Skipping {conv_id}: found voices {sorted({s for _,s,_ in items})}")
        return
    slug1, slug2 = seen

    print(f"🔄 Composing {conv_id}: slug1={slug1}, slug2={slug2}, turns={[t for t,_,_ in items]}")

    # Load each segment, skipping failures
    segments = []
    for turn, slug, fn in items:
        path = os.path.join(inp, fn)
        try:
            seg = AudioSegment.from_mp3(path)
        except CouldntDecodeError:
            print(f"⚠️  Failed to decode {fn}; skipping turn {turn}.")
            continue
        segments.append(seg)

    if not segments:
        print(f"⚠️  No valid segments for {conv_id}, skipping.")
        return

    # Stitch with 500 ms lead + 500 ms gaps
    out = AudioSegment.silent(duration=200)
    for i, seg in enumerate(segments):
        out += seg
        if i < len(segments) - 1:
            out += AudioSegment.silent(duration=200)

    # Ensure output dir exists
    os.makedirs(outp, exist_ok=True)
    out_fn = f"{conv_id}_composed_{slug1}_{slug2}.mp3"
    out.export(os.path.join(outp, out_fn), format="mp3")
    print(f"✅ Saved {out_fn}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Compose a 2-voice conversation by turn order."
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing per-turn MP3s (slug_conv_turn.mp3)"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to write composed MP3s"
    )
    parser.add_argument(
        "--conv-id", "-c", required=True,
        help="Conversation ID to process"
    )
    args = parser.parse_args()

    compose_one(args.conv_id, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

