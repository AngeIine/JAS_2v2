# JAS-2v2 Code

This directory contains the reference code release for JAS-2v2. The code supports sequence construction, feature extraction, baseline comparison, temporal permutation analysis, and the unified joint action-to-score model used in the accompanying paper.

## Contents

- `dataset.py`: unified CSV loader and sequence builder
- `models.py`: temporal graph backbone used by the lightweight multi-task scaffold
- `train.py`: lightweight unified training and evaluation entry point
- `train_joint_action_score_dag.py`: reference implementation of the joint action-to-score model
- `compare_sequence_baselines.py`: baseline comparison script
- `temporal_permutation_test.py`: temporal permutation analysis
- `extract_action_features.py`: action-level feature extraction from source videos and aligned annotations

## Environment

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Expected Layout

The code is intended to be run from the `JAS_2v2` root directory with the following layout:

```text
JAS_2v2/
  README.md
  requirements.txt
  data/
  splits/
  code/
    train.py
    train_joint_action_score_dag.py
    ...
```

## Basic Usage

### Unified sequence training

```bash
python code/train.py --data-root data --split-mode mixed --feature-tier common
```

### Joint action-to-score training

```bash
python code/train_joint_action_score_dag.py
```

### Baseline comparison

```bash
python code/compare_sequence_baselines.py
```

### Temporal permutation analysis

```bash
python code/temporal_permutation_test.py
```

### Feature extraction from source videos

```bash
python code/extract_action_features.py \
  --video path/to/video.mp4 \
  --csv path/to/action_annotations.csv \
  --output-csv path/to/processed_annotations.csv
```

This script requires aligned source videos and action annotations. For badminton and beach volleyball files that use timestamp-style temporal fields, users should first convert the annotations into frame-aligned action boundaries before reconstructing the full paper pipeline.

## Notes

- The public dataset release focuses on annotations, split files, and metadata.
- Raw source videos are not redistributed in the default release package.
- Feature extraction is therefore optional for users who only want to work with the released annotation benchmark.
