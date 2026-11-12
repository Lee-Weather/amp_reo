#!/usr/bin/env bash
# exp3.1 MuJoCo replay protocol (identical to the exp3.0 diagnostics).
# Requires a VERIFIED model_11750.pt in this directory.
set -eu
cd "$(dirname "$0")/../../.."
MODEL=czy/analysis/exp31_diag/model_11750.pt
ROOT=e:/X1/amp_re0/tmp_exp30_snapshot
MJCF=e:/X1/AMP/resources/robots/x1/mjcf
MESHES=e:/X1/amp_re0/tmp_exp30_snapshot/resources/robots/x1_29/meshes
RUNTIME=e:/X1/amp_re0/agibot_x1_train-main/czy/exp1/records/exp2.9_long_evaluation/runtime_audit/runtime.json
PROBE=e:/X1/amp_re0/agibot_x1_train-main/czy/exp1/records/exp2.9_long_evaluation/runtime_audit/passive_probe.json
COMMIT=$(git rev-parse HEAD)

run() {
  local name=$1; shift
  local out=czy/analysis/exp31_diag/$name
  python scripts/amp_tools/sim2sim_amp_mlp.py \
    --root "$ROOT" --checkpoint "$MODEL" \
    --mjcf-source "$MJCF" --meshes "$MESHES" --out-dir "$out" \
    --video-path "$out/replay.mp4" --csv-path "$out/replay.csv" \
    --training-commit "$COMMIT" --runtime-properties "$RUNTIME" --passive-probe "$PROBE" \
    "$@"
}

run G1_zero_heading --vx 0.001 --stand-seconds 0 --transition-seconds 0.02 --seconds 3
run G2_zero_nohead --vx 0.001 --stand-seconds 0 --transition-seconds 0.02 --seconds 3 --disable-heading
run G3_zero_10s --vx 0.001 --stand-seconds 0 --transition-seconds 0.02 --seconds 10 --disable-heading
run G4_buffer_step_10s --vx 0.5953696966 --stand-seconds 1.5 --transition-seconds 0.02 --seconds 10