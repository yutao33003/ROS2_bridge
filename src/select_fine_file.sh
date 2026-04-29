#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage:"
    echo "  bash src/select_fine_file.sh <root_dir> [tag]"
    echo "Examples:"
    echo "  bash src/select_fine_file.sh /data/chenp6/SegmentationTask/data/ward_dataset0421 ward1"
    echo "  bash src/select_fine_file.sh /data/chenp6/SegmentationTask/data/ward_dataset0421"
    exit 1
fi

root_dir=$1
single_tag="${2:-}"
preview_root="$root_dir/checked/ground_truth_preview"

if [ ! -d "$preview_root" ]; then
    echo "❌ preview root not found: $preview_root"
    exit 1
fi

shopt -s nullglob

process_tag() {
    local tag="$1"
    local gt_dir="$preview_root/$tag"
    local rgb_dir="$root_dir/rgb_image/${tag}_rgb"
    local out_dir="$root_dir/selected_rgb_image/${tag}_rgb"

    if [ ! -d "$gt_dir" ]; then
        echo "❌ ground-truth dir not found: $gt_dir"
        return
    fi

    if [ ! -d "$rgb_dir" ]; then
        echo "❌ rgb dir not found: $rgb_dir"
        return
    fi

    mkdir -p "$out_dir"
    local copied_count=0

    for f in "$gt_dir"/*; do
        [ -f "$f" ] || continue

        local name
        name=$(basename "$f")
        name="${name%.*}"
        name="${name%__gt}"

        local candidates=(
            "$rgb_dir/$name.jpg"
            "$rgb_dir/$name.png"
            "$rgb_dir/${name}_rgb.jpg"
            "$rgb_dir/${name}_rgb.png"
        )

        local copied=0
        local src
        for src in "${candidates[@]}"; do
            if [ -f "$src" ]; then
                cp "$src" "$out_dir/"
                copied=1
                copied_count=$((copied_count + 1))
                break
            fi
        done

        if [ "$copied" -eq 0 ]; then
            echo "❌ [$tag] missing source for preview: $f"
        fi
    done

    echo "✅ [$tag] copied $copied_count files -> $out_dir"
}

if [ -n "$single_tag" ]; then
    process_tag "$single_tag"
else
    tag_dirs=("$preview_root"/*)
    if [ "${#tag_dirs[@]}" -eq 0 ]; then
        echo "❌ no tag directories found under: $preview_root"
        exit 1
    fi

    for tag_dir in "${tag_dirs[@]}"; do
        [ -d "$tag_dir" ] || continue
        process_tag "$(basename "$tag_dir")"
    done
fi
