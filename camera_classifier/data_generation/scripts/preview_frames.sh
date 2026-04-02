#!/usr/bin/env bash
set -e

# Usage:
#   ./review_frames.sh input.mp4 output_dir [num_frames]
#
# Example:
#   ./preview_frames.sh match1.mp4 match1_frames 5000

INPUT="$1"
OUTDIR="$2"
NUM_FRAMES="${3:-5000}"

if [ -z "$INPUT" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: $0 input.mp4 output_dir [num_frames]"
    exit 1
fi

mkdir -p "$OUTDIR"

ffmpeg -y \
    -i "$INPUT" \
    -vf "fps=3,select='lt(n\,${NUM_FRAMES})'" \
    -vsync 0 \
    -start_number 0 \
    "$OUTDIR/frame_%06d.jpg"

echo "Done. Extracted first $NUM_FRAMES frames at 3 fps into $OUTDIR"
