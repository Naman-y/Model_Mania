"""
Amazon ML Challenge 2026 — Kaggle & AWS Batch Execution Runner
Run this script on Kaggle or AWS EC2 to generate the final competition outputs.

Usage on Kaggle:
    python kaggle_runner.py --data-dir /kaggle/input/amazon-ml-challenge-2026/student_resource/dataset --out-dir /kaggle/working/output

Usage on AWS:
    python kaggle_runner.py --data-dir ./student_resource/dataset --out-dir ./output
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline import run_test_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle/AWS Full Execution Runner")
    parser.add_argument("--data-dir", default="student_resource/dataset", help="Path to dataset directory")
    parser.add_argument("--out-dir", default="output", help="Directory to save matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--train-sample", type=int, default=100000, help="Number of S1 training records")
    parser.add_argument("--max-cand-sample", type=int, default=500000, help="Number of candidate records for training")
    parser.add_argument("--test-max-rows", type=int, default=None, help="None = run full 1.7M test set")
    parser.add_argument("--threshold", type=float, default=0.55, help="Classification probability cutoff")
    args = parser.parse_args()

    print(f"Starting execution with data from: {args.data_dir}")
    print(f"Output will be saved to: {args.out_dir}")
    run_test_mode(args)
    print("Execution complete! You can now zip and submit the files in the output directory.")
