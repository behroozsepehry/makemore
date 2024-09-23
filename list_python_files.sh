#!/bin/bash

echo "Here is my python codebase:"
echo -e "===========================\n"

# Function to process Python files
process_files() {
    find . -name "*.py" -type f | sort | while read -r file; do
        echo "File: ${file#./}"
        echo "Contents:"
        echo '```'
        cat "$file"
        echo '```'
        echo -e "\n-------------------\n"
    done
}

# Start processing from the current directory
process_files