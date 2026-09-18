#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

bash "$SCRIPT_DIR/run_one.sh" rank8  8  5.0e-5 4 bnb
bash "$SCRIPT_DIR/run_one.sh" rank32 32 5.0e-5 4 bnb
bash "$SCRIPT_DIR/run_one.sh" lr1e5  16 1.0e-5 4 bnb
bash "$SCRIPT_DIR/run_one.sh" lr1e4  16 1.0e-4 4 bnb
bash "$SCRIPT_DIR/run_one.sh" lora   16 5.0e-5 none none

echo "===== 全部完成 ====="
cat "$SCRIPT_DIR/../results/results.csv"
