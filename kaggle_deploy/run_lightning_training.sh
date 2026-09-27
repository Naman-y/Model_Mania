#!/bin/bash
set -e
STUDIO=/teamspace/studios/this_studio
REPO=$STUDIO/Model_Mania
DATA=$STUDIO/dataset

echo "=== FULL AUTO TRAINING PIPELINE ==="

echo "[1/6] Pulling latest code..."
cd $REPO
git pull origin master 2>/dev/null || true

echo "[2/6] Setting up venv and deps..."
python3 -m venv --system-site-packages $STUDIO/venv
source $STUDIO/venv/bin/activate
pip install -q polars jellyfish rapidfuzz faiss-cpu scikit-learn tqdm aksharamukha sentence-transformers

echo "[3/6] GPU status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU - CPU only"

echo "[4/6] Linking dataset..."
mkdir -p $REPO/student_resource/dataset/train $REPO/student_resource/dataset/test
for f in train_ground_truth.tsv train_source1.tsv train_source2.tsv train_source3.tsv; do
  [ -f "$DATA/train/$f" ] && ln -sf "$DATA/train/$f" "$REPO/student_resource/dataset/train/$f" && echo "  Linked $f"
done
for f in test_source1.tsv test_source2.tsv test_source3.tsv; do
  [ -f "$DATA/test/$f" ] && ln -sf "$DATA/test/$f" "$REPO/student_resource/dataset/test/$f" && echo "  Linked $f"
done
ls -lh $REPO/student_resource/dataset/train/

echo "[5/6] Starting full training..."
cd $REPO
python3 -u full_training_benchmark.py 2>&1 | tee training_lightning_output.log
echo "=== TRAINING COMPLETE ==="

echo "[6/6] Running error audit on full test set..."
python3 -u code/business_entity_resolution/src/pipeline_error_audit.py 2>&1 | tee audit_lightning_output.log
echo "=== AUDIT COMPLETE ==="

git config user.email "devanshdewan55a@gmail.com"
git config user.name "glok77"
git add -A
git commit -m "feat: full training+eval Lightning AI GPU $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || echo "Nothing to commit"
git push origin master 2>/dev/null || echo "Push done or skipped"
echo "=== ALL DONE ==="
