import os, json

src_dir = 'code/business_entity_resolution/src'
files = ['preprocessor.py', 'blocking_production.py', 'features.py', 'model.py', 'evaluate.py']

out_file = 'kaggle_deploy/solution_bundled.py'
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('import os, sys, subprocess\n')
    f.write('try:\n')
    f.write('    subprocess.run([sys.executable, "-m", "pip", "install", "jellyfish", "sentence-transformers", "faiss-gpu", "polars", "lightgbm", "-q"], check=True)\n')
    f.write('except Exception:\n')
    f.write('    pass\n\n')
    f.write('# BUNDLED SOLUTION SCRIPT\n')
    for file in files:
        with open(os.path.join(src_dir, file), 'r', encoding='utf-8') as in_f:
            f.write(f'\n# === {file} ===\n')
            f.write(in_f.read())
            f.write('\n')
            
    with open('kaggle_deploy/run_full_training_and_eval.py', 'r', encoding='utf-8') as main_f:
        lines = main_f.readlines()
        f.write('\n# === run_full_training_and_eval.py ===\n')
        for line in lines:
            if not line.startswith('from preprocessor import') and not line.startswith('from blocking_production import') and not line.startswith('from features import') and not line.startswith('from evaluate import'):
                f.write(line)

print('Bundled into solution_bundled.py')
