#!/usr/bin/env bash
# AWS EC2 Execution Script for Amazon ML Challenge 2026
# Sets up virtual environment, installs dependencies, and runs full test inference.

set -e

echo "=== Setting up Amazon ML Challenge Environment on AWS ==="
sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv zip

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt

echo "=== Running Full Test Inference Pipeline ==="
python3 code/business_entity_resolution/src/pipeline.py \
    --mode test \
    --data-dir student_resource/dataset \
    --out-dir output \
    --threshold 0.70

echo "=== Validating Submission Format ==="
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test

echo "=== Packaging Final Submission Zip ==="
python3 code/business_entity_resolution/src/package_submission.py --team-name "ModelMania"

echo "=== Complete! Submission package ready ==="
