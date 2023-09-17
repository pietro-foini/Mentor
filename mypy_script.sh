#!/bin/bash

# Run mypy and save the output to the "output" variable
output=$(mypy --show-error-codes --no-pretty --no-error-summary --config-file mypy.ini "$@" 2>&1)

# Redirect the stdout to a file named "mypy_recap.txt"
echo "$output" | grep -E 'error:|: note:|: warning:' | grep -v ' error: \(this|TypeVar\)' | grep -v 'warning: unused import' >> mypy_recap.txt


# Set the exit code to 0
exit 0