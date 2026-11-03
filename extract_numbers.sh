#!/bin/bash
# Print the sample standard error for each benchmark in one or more input files.
# Usage: ./extract_numbers.sh <filename> [filename ...]
# Output: benchmark label, tab, standard error (in the input timing unit).
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <filename> [filename ...]" >&2
    exit 1
fi

for file in "$@"; do
    if [ ! -f "$file" ] || [ ! -r "$file" ]; then
        echo "Error: Missing or unreadable file: $file" >&2
        exit 1
    fi
done

LC_ALL=C awk -F '\t' '
    function fail(message) {
        print "Error: " message > "/dev/stderr"
        failed = 1
        exit 1
    }

    /^[[:space:]]*$/ { next }

    {
        sub(/\r$/, "")
        if (NF != 2 || $1 == "" || $2 !~ /^[0-9]+([.][0-9]+)?( ns)?$/) {
            fail(FILENAME ":" FNR ": expected a label, a tab, and a numeric timing (optionally followed by ns)")
        }

        label = $1
        value = $2 + 0

        if (!(label in seen)) {
            seen[label] = 1
            labels[++label_count] = label
        }

        # The Welford algorithm avoids loss of precision for large timings.
        counts[label]++
        delta = value - means[label]
        means[label] += delta / counts[label]
        m2[label] += delta * (value - means[label])
    }

    END {
        if (failed) exit 1
        if (label_count == 0) {
            fail("no observations found")
        }

        for (i = 1; i <= label_count; i++) {
            label = labels[i]
            if (counts[label] < 2) {
                fail("standard error for " label " requires at least two observations")
            }

            # SEM = sample standard deviation / sqrt(sample count).
            standard_error = sqrt(m2[label] / (counts[label] - 1)) / sqrt(counts[label])
            printf "%s\t%.6f\n", label, standard_error
        }
    }
' "$@"
