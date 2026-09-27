#!/usr/bin/env python
"""Explore dataset structure and characteristics."""

import polars as pl
import os

os.chdir('student_resource/dataset/train')

# ===== SOURCE 1 =====
print('=' * 100)
print('SOURCE 1 (S1) - Company Database Snapshot')
print('=' * 100)
s1 = pl.read_csv('train_source1.tsv', separator='\t', n_rows=5, encoding='utf-8-sig')
print(s1)
s1_full = pl.read_csv('train_source1.tsv', separator='\t', encoding='utf-8-sig')
print(f'\n✓ Total S1 records: {len(s1_full):,}')
print(f'✓ Columns: {s1.columns}')

# Show some real examples
print('\n📊 S1 Sample Records (with details):')
for i, row in enumerate(s1_full.head(3).iter_rows(named=True)):
    print(f"\n  Record {i+1}:")
    print(f"    Entity ID: {row['entity_id']}")
    print(f"    Name: {row['business_name']}")
    print(f"    Address: {row['business_address'][:60]}...")
    print(f"    Country: {row['country']}")

# ===== SOURCE 2 =====
print('\n' + '=' * 100)
print('SOURCE 2 (S2) - Company Database Snapshot')
print('=' * 100)
s2 = pl.read_csv('train_source2.tsv', separator='\t', n_rows=5, encoding='utf-8-sig')
print(s2)
s2_full = pl.read_csv('train_source2.tsv', separator='\t', encoding='utf-8-sig')
print(f'\n✓ Total S2 records: {len(s2_full):,}')

# ===== SOURCE 3 =====
print('\n' + '=' * 100)
print('SOURCE 3 (S3) - Company Database Snapshot')
print('=' * 100)
s3 = pl.read_csv('train_source3.tsv', separator='\t', n_rows=5, encoding='utf-8-sig')
print(s3)
s3_full = pl.read_csv('train_source3.tsv', separator='\t', encoding='utf-8-sig')
print(f'\n✓ Total S3 records: {len(s3_full):,}')

# ===== GROUND TRUTH =====
print('\n' + '=' * 100)
print('GROUND TRUTH - Match Pairs')
print('=' * 100)
gt = pl.read_csv('train_ground_truth.tsv', separator='\t', n_rows=10, encoding='utf-8-sig')
print(gt)
gt_full = pl.read_csv('train_ground_truth.tsv', separator='\t', encoding='utf-8-sig')
print(f'\n✓ Total GT pairs: {len(gt_full):,}')
print(f'✓ Columns: {gt.columns}')

# Show example match
print('\n📊 Example GT Pair:')
example = gt_full.head(1).to_dicts()[0]
print(f"  S1 Entity: {example['source1_entity_id']}")
print(f"  Matches: {example['matched_entity_ids']}")

# ===== COUNTRY DISTRIBUTION =====
print('\n' + '=' * 100)
print('🌍 COUNTRY DISTRIBUTION')
print('=' * 100)
print('\nS1 Countries (top 15):')
s1_countries = s1_full['country'].value_counts().sort('count', descending=True).head(15)
print(s1_countries)

print('\nS2 Countries (top 15):')
s2_countries = s2_full['country'].value_counts().sort('count', descending=True).head(15)
print(s2_countries)

print('\nS3 Countries (top 15):')
s3_countries = s3_full['country'].value_counts().sort('count', descending=True).head(15)
print(s3_countries)

# ===== DATA QUALITY =====
print('\n' + '=' * 100)
print('📋 DATA QUALITY CHECKS')
print('=' * 100)

print('\nS1 Missing Values:')
for col in s1_full.columns:
    null_count = s1_full[col].null_count()
    if null_count > 0:
        pct = 100 * null_count / len(s1_full)
        print(f"  {col}: {null_count:,} ({pct:.2f}%)")

print('\nS2 Missing Values:')
for col in s2_full.columns:
    null_count = s2_full[col].null_count()
    if null_count > 0:
        pct = 100 * null_count / len(s2_full)
        print(f"  {col}: {null_count:,} ({pct:.2f}%)")

print('\nS3 Missing Values:')
for col in s3_full.columns:
    null_count = s3_full[col].null_count()
    if null_count > 0:
        pct = 100 * null_count / len(s3_full)
        print(f"  {col}: {null_count:,} ({pct:.2f}%)")

# ===== NAME CHARACTERISTICS =====
print('\n' + '=' * 100)
print('📝 NAME CHARACTERISTICS')
print('=' * 100)

print('\nS1 Name Length Distribution:')
s1_name_lens = s1_full['business_name'].str.len_chars().to_list()
print(f"  Min: {min(s1_name_lens):,} chars")
print(f"  Max: {max(s1_name_lens):,} chars")
print(f"  Mean: {sum(s1_name_lens)/len(s1_name_lens):.1f} chars")
print(f"  Median: {sorted(s1_name_lens)[len(s1_name_lens)//2]:,} chars")

print('\nS2 Name Length Distribution:')
s2_name_lens = s2_full['business_name'].str.len_chars().to_list()
print(f"  Min: {min(s2_name_lens):,} chars")
print(f"  Max: {max(s2_name_lens):,} chars")
print(f"  Mean: {sum(s2_name_lens)/len(s2_name_lens):.1f} chars")
print(f"  Median: {sorted(s2_name_lens)[len(s2_name_lens)//2]:,} chars")

print('\nS3 Name Length Distribution:')
s3_name_lens = s3_full['business_name'].str.len_chars().to_list()
print(f"  Min: {min(s3_name_lens):,} chars")
print(f"  Max: {max(s3_name_lens):,} chars")
print(f"  Mean: {sum(s3_name_lens)/len(s3_name_lens):.1f} chars")
print(f"  Median: {sorted(s3_name_lens)[len(s3_name_lens)//2]:,} chars")

# ===== TRANSLITERATION & SPECIAL CHARS =====
print('\n' + '=' * 100)
print('🌐 TRANSLITERATION & SPECIAL CHARACTERS')
print('=' * 100)

def check_unicode(df, name=''):
    non_ascii = 0
    for text in df['business_name'].to_list():
        if any(ord(c) > 127 for c in str(text)):
            non_ascii += 1
    pct = 100 * non_ascii / len(df) if len(df) > 0 else 0
    print(f"  {name} has non-ASCII: {non_ascii:,} / {len(df):,} ({pct:.2f}%)")

check_unicode(s1_full, 'S1')
check_unicode(s2_full, 'S2')
check_unicode(s3_full, 'S3')

# Show some non-ASCII examples
print('\nExamples of non-ASCII names:')
for i, row in enumerate(s1_full.filter(
    pl.col('business_name').str.contains('[^\x00-\x7F]')
).head(5).iter_rows(named=True)):
    print(f"  {row['entity_id']}: {row['business_name']}")

print('\n' + '=' * 100)
