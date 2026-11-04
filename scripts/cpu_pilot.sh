#!/bin/bash
# CPU pipeline pilot: all three adaptation methods on 10 hard CROHME 2014 images.
# Not a personalization result: the support images have no writer IDs.
# It checks that each method can fit a tiny support set (guide stage 2), what it
# does to held-out images, and what each method costs. Run from the repo root
# after data/pilot/support_10.jsonl exists (see docs/eval-log.md).
set -euo pipefail
PY=${PY:-.venv/bin/python}
STEPS=${STEPS:-200}
OUT=runs/pilot_cpu

# Held-out sets: the other 50 screened 2014 images, and 200 fixed 2019 images.
$PY - <<'EOF'
import json, random
support = {json.loads(l)["id"] for l in open("data/pilot/support_10.jsonl")}
screen = [l for l in open("data/pilot/screen_2014.jsonl") if json.loads(l)["id"] not in support]
open("data/pilot/heldout_2014_50.jsonl", "w").writelines(screen)
rows = open("data/general_2019.jsonl").readlines()
random.Random(0).shuffle(rows)
open("data/pilot/general_2019_200.jsonl", "w").writelines(rows[:200])
EOF

evaluate() {  # checkpoint, manifest, output, extra flags
  $PY -m wlm.predict evaluate --checkpoint "$1" --manifest "$2" --output "$3" \
    --kind general --device cpu "${@:4}" > "$3.log" 2>&1
}
mkdir -p $OUT/base
evaluate checkpoints/base.pt data/pilot/support_10.jsonl $OUT/base/support --as-base
evaluate checkpoints/base.pt data/pilot/heldout_2014_50.jsonl $OUT/base/heldout_2014 --as-base
evaluate checkpoints/base.pt data/pilot/general_2019_200.jsonl $OUT/base/general_2019 --as-base

for method in lora decoder full; do
  $PY -m wlm.train --base checkpoints/base.pt --manifest data/pilot/support_10.jsonl \
    --method $method --steps $STEPS --batch-size 4 --seed 0 \
    --output $OUT/$method --device cpu > $OUT/$method.train.log 2>&1
  for split in support heldout_2014 general_2019; do
    case $split in
      support) manifest=data/pilot/support_10.jsonl ;;
      heldout_2014) manifest=data/pilot/heldout_2014_50.jsonl ;;
      general_2019) manifest=data/pilot/general_2019_200.jsonl ;;
    esac
    evaluate $OUT/$method/model.pt $manifest $OUT/$method/$split
  done
done
echo "Pilot finished: $OUT"
