#!/usr/bin/env bash
#
# MVTracker on 100 tracks per trial: a random subset of MVTracker's cached tracks (pose3d_cached_100.npz,
# scripts/make_pose3d_cached_100.py), on the canonical cameras, with the paper-protocol metrics. This is
# mvtracker_cached.sh (query_first, MVTracker's official per-dataset settings and scene normalization, DUSt3R and
# ground-truth depths, linking of the canonical depths) run on the 100-track files; see there for the steps.
# Results: $OUTPUT_BASE/paper-canonical-cached-100-duster and ...-gt
#
# Usage:
#   bash mvtracker_100.sh [canonical] [DATASETS] [POSE_FILE]
#
# Examples:
#   bash mvtracker_100.sh
#   bash mvtracker_100.sh canonical "dex_ycb cmupanoptic_3dgs"
#
# Env overrides: DEPTH_TYPES (default "duster gt"), OUTPUT_BASE, DEVICE, PYTHON (as in mvtracker_cached.sh).
# A number of random cameras instead of canonical is not meaningful for the cached tracks, whose visibility is only
# given for the canonical cameras.

set -e

MODE=${1:-canonical}
DATASETS=${2:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${3:-pose3d_cached_100.npz}

if [[ "$MODE" != "canonical" ]]; then
    echo "only 'canonical' is supported for the cached tracks, got '$MODE'"; exit 1
fi

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_BASE="${OUTPUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/mvtracker-outputs}"
export OUTPUT_TAG=cached-100

bash "$SCRIPTS/mvtracker_cached.sh" "$DATASETS" "$POSE_FILE" "${DEPTH_TYPES:-duster gt}"
