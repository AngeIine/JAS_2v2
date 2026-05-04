# JAS-2v2

JAS-2v2 is a benchmark for joint action prediction and score outcome prediction in representative 2v2 sports. The current release contains action-level CSV annotations, split definitions, and dataset documentation for four sports:

- beach volleyball
- badminton doubles
- tennis doubles
- table tennis doubles

The benchmark is designed to study fine-grained tactical processes rather than only final match statistics. In the accompanying paper, each rally is modeled as an action sequence, and the task is to predict both the current action category and the sequence-level score outcome from historical action context.

## Directory Structure

```text
JAS_2v2/
  README.md
  LICENSE.txt
  metadata.json
  requirements.txt
  code/
    *.py
    README.md
  data/
    badminton/
      *.csv
    beachvolleyball/
      *.csv
    tennis/
      *.csv
    table_tennis/
      *.csv
  splits/
    train_files.txt
    test_files.txt
```

## Release Contents

This release includes:

- action-level annotation CSV files for all four sports
- rally-level score labels
- acting-player position labels
- file-level split definitions for reproducibility
- a reference code package for feature extraction and sequence modeling

## Code Release

The `code/` directory contains the reference implementation associated with the benchmark, including sequence construction, feature extraction, baseline comparison, and the joint action-to-score model described in the paper. Dependency installation is documented in `requirements.txt`, and code-specific usage notes are provided in `code/README.md`.


## Benchmark Tasks

JAS-2v2 supports two prediction tasks:

1. **Action prediction**: predict the current action category from preceding action history.
2. **Score prediction**: predict the sequence-level score outcome from the same historical context.

In the paper, these two tasks are jointly modeled under a unified action-to-score framework.

## Data Format

Each CSV file stores temporally ordered action annotations from a match or match segment. The exact field names are not completely identical across sports, because the release preserves the original sport-specific annotation format.

### Common semantic fields

The following semantic fields are used by the benchmark:

- `action_name` or `name`: action category label
- `position`: acting player position
- `score`: binary score outcome label

### Temporal fields

Two temporal field conventions appear in the release:

- `start_frame`, `end_frame`: frame indices used by the tennis and table tennis files
- `start`, `end`: timestamp strings used by the badminton and beach volleyball files

If you want to use the released annotations in a unified sequence-modeling pipeline, you should first convert all sport files into a unified frame-based format, or preprocess them into a common schema containing at least:

```text
start_frame, end_frame, action_name, position, score
```

### Optional physical-feature fields

Processed files may additionally include the derived physical features used in the paper:

- `target_distance` (TD)
- `pdistance` (PD)
- `pspeed` (PV)
- `bspeed` (BV)

Depending on preprocessing, auxiliary fields such as `duration`, `num_frames`, `target_x`, and `target_y` may also be included.

## Dataset Semantics

Each CSV file contains an ordered sequence of annotated actions. A rally sequence is constructed from consecutive actions, typically from serve to the end of the point.

### Sequential context

In the paper, model input is formed from a fixed-length action history. Let:

- $m$ denote the input sequence length
- $T_{i,k}$ denote the current action
- $\{T_{i,k-m}, \dots, T_{i,k-1}\}$ denote the preceding action context

For each target step, the benchmark uses the preceding action history as context and predicts:

- the current action category
- the sequence-level score outcome

### Semantic annotations

The benchmark provides semantic annotations that can be used to derive:

- action category embeddings
- player-position embeddings
- sport-identity embeddings for unified multi-sport modeling

These semantic signals can be combined with action-level physical features when available.

## Feature Definitions

In addition to semantic labels, the paper defines four action-level physical features:

- **Ball Velocity (BV)**: the average speed of the ball during the current action segment
- **Player Velocity (PV)**: the average motion speed of the acting player during the current action segment
- **Target Distance (TD)**: the spatial relation between the predicted landing point of the current action and the position of the player who performs the subsequent action
- **Player Distance (PD)**: the movement distance of the acting player during the current action segment

These four quantities form the physical feature vector

```text
f_c = [BV, PV, TD, PD]
```

When such fields are present in the processed CSV files, they can be directly used as derived action-level descriptors. When they are absent, users may reconstruct them from the source videos and aligned action annotations.

## Relation to Source Videos

The public release focuses on annotations and split files. The full physical feature extraction pipeline described in the paper additionally requires access to the corresponding raw videos or source clips.

If users wish to reconstruct the full paper pipeline from raw data, they should align the action annotations with the source videos, estimate ball and player trajectories, and then compute BV, PV, TD, and PD for each action segment. The public release itself is intended to describe the benchmark and its annotations; method code and training scripts will be distributed separately.

## Split Files

The `splits/` directory contains:

- `train_files.txt`
- `test_files.txt`

These files provide release-time split definitions for reproducibility and hosting. If you maintain a different official benchmark split for the final benchmark release, you should update these files accordingly.

## Intended Uses

JAS-2v2 is intended for research on:

- action-based tactical analysis
- joint action and score prediction
- multi-sport sequence modeling
- action-to-score reasoning
- structured sports intelligence benchmarks


## Contact

For questions about the release package, benchmark usage, or licensing details, please contact the dataset authors through the paper submission contact channel.
