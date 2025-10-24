#!/bin/bash

SSH_PORT="25922"
SSH_HOST="root@ssh8.vast.ai"
REMOTE_DIR="/opt/workspace-internal/DeepSeek-OCR-vllm"
LOCAL_DIR="/mnt/wsl/projects/deepseek-ocr/DeepSeek-OCR-master/DeepSeek-OCR-vllm"

ESSENTIAL_FILES=(
    "vast_setup.py"
    "vast_server.py"
    "deepseek_ocr.py"
    "config.py"
    "requirements.txt"
    "run_dpsk_ocr_image.py"
    "run_dpsk_ocr_pdf.py"
    "run_dpsk_ocr_eval_batch.py"
    "README_SERVER.md"
    "VAST_AI_SETUP.md"
)

ESSENTIAL_DIRS=(
    "process"
    "deepencoder"
)

ssh -p $SSH_PORT $SSH_HOST "mkdir -p $REMOTE_DIR"

for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        scp -P $SSH_PORT "$LOCAL_DIR/$file" $SSH_HOST:"$REMOTE_DIR/"
    fi
done

for dir in "${ESSENTIAL_DIRS[@]}"; do
    if [ -d "$LOCAL_DIR/$dir" ]; then
        scp -r -P $SSH_PORT "$LOCAL_DIR/$dir" $SSH_HOST:"$REMOTE_DIR/"
    fi
done

ssh -p $SSH_PORT $SSH_HOST "chmod +x $REMOTE_DIR/vast_setup.py $REMOTE_DIR/vast_server.py"

ssh -p $SSH_PORT $SSH_HOST "cd $REMOTE_DIR && python vast_setup.py"

ssh -p $SSH_PORT $SSH_HOST "cd $REMOTE_DIR && python vast_server.py"