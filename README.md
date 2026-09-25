# Business Entity Resolution Challenge

Build a system that links business records across three independent data sources. Source 1 is the reference set; for each Source 1 entity, predict all matching records in Sources 2 and 3. Records may have noisy names and addresses, and a Source 1 entity can have no matches, one match, or multiple matches.

## Challenge overview

Business names may contain abbreviations, legal suffix differences, trade names, typos, punctuation changes, and word-order differences. Addresses may be incomplete or formatted differently, include transliterations or landmarks, or omit components. Entity IDs identify their source (`S1-`, `S2-`, or `S3-`).

The training data covers the US and India. The test data also includes France, which is absent from training. Treat country values as open-set strings and include every test Source 1 entity in the output.

## Data

All data files use tab-separated values (TSV). Read them with an explicit tab delimiter, for example:

```python
import pandas as pd

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Training files:

- `dataset/train/train_source1.tsv` — deduplicated Source 1 records
- `dataset/train/train_source2.tsv` — Source 2 records
- `dataset/train/train_source3.tsv` — Source 3 records
- `dataset/train/train_ground_truth.tsv` — Source 1 IDs and comma-separated matching Source 2/3 IDs

Test files:

- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

Each source record file contains `entity_id`, `business_name`, `business_address`, and `country` columns. Ground truth contains `source1_entity_id` and `matched_entity_ids`.

## Submission outputs

Place both files in `output/`:

- `matching_results.tsv` — final predicted matches, with columns `source1_entity_id` and `matched_entity_ids`.
- `candidate_pairs.tsv` — the candidate set passed to the final matching model, with columns `source1_entity_id` and `candidate_entity_ids`.

Both files must be tab-separated. Include exactly one row for every Source 1 test entity, including entities with no matches. Leave the ID-list field empty for no matches. ID lists are comma-separated, must not contain duplicates, and may contain only existing Source 2 or Source 3 test IDs. Every final match must also appear in that entity's candidate list.

Example (`matching_results.tsv`):

```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

Example (`candidate_pairs.tsv`):

```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

## Method and evaluation

Use a validation split from the training data to assess performance. The challenge scores macro-averaged per-entity F<sub>0.5</sub>, which weights precision more heavily than recall. Correctly predicting no matches for a singleton scores 1.0; predicting any match for it scores 0.0.

The challenge requires a filled methodology document describing the approach, candidate generation/blocking, model and features, and other relevant details. The final submission package also requires a runnable copy of the pipeline, its dependencies, and both output files. See the challenge instructions and provided `Documentation_template.md` for the complete package requirements.

## Validate outputs

The challenge provides a standard-library validation script. From the challenge's `student_resource/` directory, run:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The validator checks output structure and references; it does not calculate the score.

## Reproducibility

This repository currently contains the challenge description only. Add the solution code, dependency file, exact commands to generate both TSV outputs, and the completed methodology document before treating this README as a complete guide to reproducing a submission.

## Fair play

Use only the provided challenge data. External business identity lookups, entity resolution services, geocoding APIs, and external data augmentation are prohibited by the challenge rules. The final model must meet the challenge's MIT or Apache 2.0 license and up-to-8-billion-parameter constraint.
