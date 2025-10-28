#!/usr/bin/env bash
set -uo pipefail

### CONFIGURATION & HELP ###

NUM_SAMPLES=50
JOBS=4
CSV_FILE="./mentalmanip_con.csv"
OUTPUT_DIR="./samples"
FINAL_DIR="$OUTPUT_DIR/0527_500_out_of_610_samples"
LOG_DIR="./logs"
API_KEY="sk_7625b241b028c94cbdc22332e98547d609576805d9f386d2"
SCRIPT="./gen_sample_11_lab.py"
MANIPULATIVE=0    # Default: manipulative samples

usage() {
  cat <<EOF
Usage: $0 [-n NUM] [-j JOBS] [-o OUTPUT_DIR] [-k API_KEY] [-M LABEL]
Generate TTS for up to NUM conversations (default $NUM_SAMPLES),
in parallel (default $JOBS jobs), for Manipulative=LABEL (default 1).

Options:
  -n NUM         Maximum number of conversations to process
  -j JOBS        Number of parallel jobs
  -o OUTPUT_DIR  Base output directory (default $OUTPUT_DIR)
  -k API_KEY     ElevenLabs API key (overrides env ELEVEN_LABS_API_KEY)
  -M LABEL       Manipulative label (0 for non-manipulative, 1 for manipulative, default 1)
  -h             Show this help message
EOF
  exit
}

while getopts "n:j:o:k:M:h" opt; do
  case $opt in
    n) NUM_SAMPLES=$OPTARG ;;
    j) JOBS=$OPTARG       ;;
    o) OUTPUT_DIR=$OPTARG ;;
    k) API_KEY=$OPTARG    ;;
    M) MANIPULATIVE=$OPTARG ;;
    h) usage              ;;
    *) usage              ;;
  esac
done

# Prerequisites
if [[ -z "${API_KEY:-}" ]]; then
  echo "❌ ERROR: No API key supplied (-k) or in \$ELEVEN_LABS_API_KEY" >&2
  exit 1
fi
if [[ ! -f "$CSV_FILE" ]]; then
  echo "❌ ERROR: CSV file not found at $CSV_FILE" >&2
  exit 1
fi
if ! command -v csvcut &>/dev/null || ! command -v csvgrep &>/dev/null; then
  echo "❌ ERROR: csvkit (csvcut/csvgrep) is required. Install via 'pip install csvkit' and retry." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$FINAL_DIR" "$LOG_DIR"

### BUILD LIST OF CONVERSATION IDs (for given Manipulative label) ###

mapfile -t ALL_CONV < <(
  csvcut -c ID,Manipulative "$CSV_FILE" \
    | csvgrep -c Manipulative -m "$MANIPULATIVE" \
    | tail -n +2 \
    | cut -d, -f1 \
    | sort -u
)

### DETERMINE WHICH IDs ARE ALREADY DONE ###

declare -A DONE
while IFS= read -r path; do
  fname=$(basename "$path")
  conv=${fname#*_}       # strip speaker_
  conv=${conv%%_*}       # up to next _
  DONE["$conv"]=1
done < <(ls -1 "$FINAL_DIR"/*.mp3 2>/dev/null || :)

### FILTER TO PROCESS ###

TO_PROCESS=()
for c in "${ALL_CONV[@]}"; do
  [[ ${DONE[$c]:-0} -eq 1 ]] && continue
  TO_PROCESS+=("$c")
done

echo "ℹ️  Total convos (Manipulative=$MANIPULATIVE): ${#ALL_CONV[@]}"
echo "✅ Already done: ${#DONE[@]}"
echo "▶ To process: ${#TO_PROCESS[@]} (capped at $NUM_SAMPLES)"

### WORKER FUNCTION ###

worker() {
  conv_id=$1
  logfile="$LOG_DIR/$conv_id.log"
  {
    echo "--- Processing $conv_id ---"
    if python "$SCRIPT" -CONV "$conv_id" -CSV "$CSV_FILE" -k "$API_KEY" -o "$OUTPUT_DIR" -M "$MANIPULATIVE"; then
      echo "SUCCESS: $conv_id"
    else
      echo "FAIL:    $conv_id"
    fi
  } &>>"$logfile"
}

export -f worker
export SCRIPT CSV_FILE API_KEY OUTPUT_DIR LOG_DIR MANIPULATIVE

### RUN IN PARALLEL ###

printf "%s\n" "${TO_PROCESS[@]:0:$NUM_SAMPLES}" \
  | xargs -P "$JOBS" -n1 -I{} bash -c 'worker "$@"' _ {}

### SUMMARY ###

succ=$(grep -rl "SUCCESS" "$LOG_DIR" | wc -l)
fail=$(grep -rl "FAIL"    "$LOG_DIR" | wc -l)
skip=${#DONE[@]}
done=$((succ + fail + skip))

echo
echo "===== SUMMARY ====="
echo "Requested:   $NUM_SAMPLES"
echo "Processed:   $done"
echo "  ✔ Success: $succ"
echo "  ✖ Fail:    $fail"
echo "  ⏭ Skipped: $skip"
echo "Logs under $LOG_DIR/*.log"
echo "==================="

