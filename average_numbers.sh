#!/bin/bash
# Average benchmark timings from 0.txt through 9.txt, preserving row order.
# Usage: ./average_numbers.sh [data_directory] > averages.txt
# Output: benchmark label, tab, mean in nanoseconds with one decimal place.
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [data_directory]" >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${1:-$script_dir/infosys_raw_data}"
files=()
for run in {0..9}; do
    file="$data_dir/$run.txt"
    if [ ! -f "$file" ] || [ ! -r "$file" ] || [ ! -s "$file" ]; then
        echo "Error: Missing, unreadable, or empty file: $file" >&2
        exit 1
    fi
    files+=("$file")
done

LC_ALL=C awk -F '\t' '
    function fail(message) {
        print "Error: " message > "/dev/stderr"
        failed = 1
        exit 1
    }
    {
        sub(/\r$/, "")
        if (NF != 2 || $1 == "" || $2 !~ /^[0-9]+( ns)?$/) {
            fail(FILENAME ":" FNR ": expected a label, a tab, and a timing (optionally followed by ns)")
        }
        if (FILENAME == ARGV[1]) {
            labels[FNR] = $1
            rows = FNR
        } else if (FNR > rows || $1 != labels[FNR]) {
            fail(FILENAME ":" FNR ": benchmark labels or ordering differ from 0.txt")
        }
        sums[FNR] += $2 + 0
        counts[FNR]++
    }
    END {
        if (failed) exit 1
        for (row = 1; row <= rows; row++) {
            if (counts[row] != 10) {
                fail("expected 10 timings for " labels[row] ", got " counts[row])
            }
        }
        for (row = 1; row <= rows; row++) {
            printf "%s\t%.1f\n", labels[row], sums[row] / 10
        }
    }
' "${files[@]}"
