#!/usr/bin/env bash
set -e

# Usage:
#   ./extract_frames.sh frame_dir kickoff_frame [num_frames]
#
# Example:
#   ./extract_frames.sh match1_frames 215 1000

FRAMEDIR="$1"
KICKOFF="$2"
NUM_FRAMES="${3:-1000}"

if [ -z "$FRAMEDIR" ] || [ -z "$KICKOFF" ]; then
    echo "Usage: $0 frame_dir kickoff_frame [num_frames]"
    exit 1
fi

if [ ! -d "$FRAMEDIR" ]; then
    echo "Error: directory '$FRAMEDIR' does not exist"
    exit 1
fi

TMPDIR="${FRAMEDIR}_tmp_keep"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

for ((i=0; i<NUM_FRAMES; i++)); do
    SRC_NUM=$((KICKOFF + i))
    SRC_FILE=$(printf "%s/frame_%06d.png" "$FRAMEDIR" "$SRC_NUM")
    DST_FILE=$(printf "%s/frame_%06d.png" "$TMPDIR" "$i")

    if [ ! -f "$SRC_FILE" ]; then
        echo "Error: missing source frame $SRC_FILE"
        echo "You probably did not extract enough preview frames."
        rm -rf "$TMPDIR"
        exit 1
    fi

    mv "$SRC_FILE" "$DST_FILE"
done

# Delete everything else from the original frame directory
find "$FRAMEDIR" -maxdepth 1 -type f -name 'frame_*.png' -delete

# Move the kept/renumbered files back into the original directory
mv "$TMPDIR"/frame_*.png "$FRAMEDIR"/
rmdir "$TMPDIR"

echo "Done. Kept $NUM_FRAMES frames starting at kickoff frame $KICKOFF in $FRAMEDIR"