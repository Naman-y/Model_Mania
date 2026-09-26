"""
Official Submission Packaging Utility for Amazon ML Challenge 2026.
Creates the final required zip archive:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import zipfile
import argparse
import subprocess
import sys

def package_submission(team_name: str, base_dir: str = "."):
    zip_name = f"{team_name}_submission.zip"
    print(f"Creating submission package: {zip_name}...")
    
    output_match = os.path.join(base_dir, "output", "matching_results.tsv")
    output_cand = os.path.join(base_dir, "output", "candidate_pairs.tsv")
    doc_template = os.path.join(base_dir, "Documentation_template.md")
    code_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    
    # Verify required files exist
    for p in [output_match, output_cand, doc_template, code_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required submission component: {p}")
            
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(output_match, arcname="output/matching_results.tsv")
        zf.write(output_cand, arcname="output/candidate_pairs.tsv")
        
        # 2. Documentation template
        zf.write(doc_template, arcname="Documentation_template.md")
        
        # 3. Code folder
        for root, dirs, files in os.walk(code_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, base_dir).replace("\\", "/")
                zf.write(full_p, arcname=rel_p)
                
    print(f"Submission zip successfully created at: {zip_name} ({os.path.getsize(zip_name):,} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip archive")
    parser.add_argument("--team-name", default="ModelMania", help="Team name for zip prefix")
    args = parser.parse_args()
    package_submission(args.team_name)
