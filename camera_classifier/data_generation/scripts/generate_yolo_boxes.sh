ROOT="/home/travis/Projects/football_event_data_generation/camera_classifier/data_generation/frame_data"
# WEIGHTS="/home/travis/Projects/yolo_all/yolov9/ext_weights/yolov9-e.pt"
WEIGHTS="/home/travis/Projects/yolo_all/yolov9/ext_weights/yolov9-c.pt"
PYTHON="/home/travis/Projects/yolo_all/yolov9env/bin/python3"
YOLO_ROOT="/home/travis/Projects/yolo_all/yolov9"
SCRIPT="$YOLO_ROOT/batch_detect.py"

# DIRS=(
#     "AC_Mil_vs_Pisa"
#     "Ars_ManCit"
#     "Brent_vs_Nott"
#     "Burn_vs_Ful"
#     "Fior_Cagl"
#     "Ham_StPaul"
#     "Hoff_vs_Leip"
#     "Leg_VallD"
#     "Lev_vs_Elch"
#     "Mon_vs_Nic"
#     "Stras_Aux"
#     "frames_for_travis/bayern_frames_3fps"
#     "frames_for_travis/como_frames_3fps"
#     "frames_for_travis/OM_frames_3fps"
#     "frames_for_travis/sevilla_frames_30"
# )
DIRS=("Sev_Bil")

for rel in "${DIRS[@]}"; do
    d="$ROOT/$rel"

    [ -d "$d" ] || {
        echo "Skipping missing directory: $d"
        continue
    }

    name="$rel"
    if [[ "$rel" == frames_for_travis/* ]]; then
        name="${rel#frames_for_travis/}"
    fi

    (
        cd "$YOLO_ROOT"
        "$PYTHON" "$SCRIPT" \
            --weights "$WEIGHTS" \
            --source "$d" \
            --project /home/travis/Projects/football_event_data_generation/camera_classifier/data_generation/scripts/runs/boxes_only \
            --name "$name" \
            --batch-size 8 \
            --exist-ok \
            --classes 0
    )
done