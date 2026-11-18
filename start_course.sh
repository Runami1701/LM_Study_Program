#!/usr/bin/env sh
# Start the course (Linux/macOS): ./start_course.sh  (or double-click in a file manager that runs scripts)
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
    exec python3 start_course.py
else
    exec python start_course.py
fi
