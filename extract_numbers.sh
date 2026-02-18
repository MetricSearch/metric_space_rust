#!/bin/bash

# Check if a filename was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

file="$1"

# Check if file exists
if [ ! -f "$file" ]; then
    echo "Error: File not found."
    exit 1
fi

# Extract the first whitespace-separated field after the tab
while IFS=$'\t' read -r _ rest; do
    # Take only the first space-separated token
    number="${rest%% *}"
    
    if [ -n "$number" ]; then
        echo "$number"
    fi
done < "$file"