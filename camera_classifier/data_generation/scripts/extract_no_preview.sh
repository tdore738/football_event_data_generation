#!/usr/bin/env bash
set -e

# Usage:
#   ./extract_no_preview.sh input.mp4 output_dir [num_frames] [frame_start]
#
# Example:
#   ./extract_no_preview.sh match1.mp4 match1_frames 1000 5000

INPUT="$1"
OUTDIR="$2"
NUM_FRAMES="${3:-1000}"
FRAME_START="${4:-5000}"

if [ -z "$INPUT" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: $0 input.mp4 output_dir [num_frames]"
    exit 1
fi

mkdir -p "$OUTDIR"

ffmpeg -y \
    -i "$INPUT" \
    -vf "fps=3,select='between(n\,${FRAME_START}\,$((${FRAME_START}+${NUM_FRAMES}-1)))'" \
    -vsync 0 \
    -start_number 0 \
    "$OUTDIR/frame_%06d.jpg"

echo "Done. Extracted $NUM_FRAMES frames starting from $FRAME_START at 3 fps into $OUTDIR"
