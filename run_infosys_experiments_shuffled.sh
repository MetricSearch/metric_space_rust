#!/bin/bash
# Run the random_access examples ten times, saving timings separately from the
# original experiments in infosys_raw_data_shuffled/0.txt through 9.txt.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"

OUT_DIR="$script_dir/infosys_raw_data_shuffled"
mkdir -p -- "$OUT_DIR"
printf '%s\n' "$OUT_DIR"

examples=(
    evp_100dim_brute_force_example_shuffled
    evp_384dim_brute_force_example_shuffled
    evp_500dim_brute_force_example_shuffled
    evp_768dim_brute_force_example_shuffled
    4bit_brute_force_example_shuffled
    8bit_brute_force_example_shuffled
    16bit_brute_force_example_shuffled
    f32_euc_brute_force_example_shuffled
    hamming_100dim_brute_force_example_shuffled
    hamming_384dim_brute_force_example_shuffled
    hamming_500dim_brute_force_example_shuffled
    hamming_768dim_brute_force_example_shuffled
)

for LOOP in {0..9}; do
    {
        for example in "${examples[@]}"; do
            cargo run -r -p benchmarking --example "$example"
        done
    } > "$OUT_DIR/$LOOP.txt"
done
