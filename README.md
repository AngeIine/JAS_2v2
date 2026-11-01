<div align="center">

# JAS-2v2

### Beyond Score: A Dataset for Joint Action and Score Predictions in 2v2 Sports

**NeurIPS 2026 · Evaluations and Datasets**

Yuting Wang · Peng Wen · Fengshun Wang · Jiayong Zuo · Xinyu Peng · Qiurui Wang

[Dataset](data) · [Code](code) · [Splits](splits) · [Getting Started](#getting-started) · [Citation](#citation)

</div>

## Overview

**JAS-2v2** is a multi-sport dataset and benchmark for **joint action prediction and score outcome prediction**. It brings together **beach volleyball, badminton doubles, tennis doubles, and table tennis doubles** in a shared action-sequence modeling framework.

The benchmark shifts attention from aggregate match statistics to the development of individual rallies: the actions performed, their temporal organization, and their relationship to the final rally outcome. It supports research on action-level representations, structured relational modeling, and multi-sport sports analysis.

<p align="center">
  <img src="assets/figures/dataset_overview.png" alt="JAS-2v2 overview showing four sports and their action taxonomies" width="900">
</p>

*Figure 2 from the paper: the four sports and their action taxonomies under the JAS-2v2 framework.*

### At a glance

| Sports | Matches | Rally sequences | Action instances | Action classes |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 47 | 4,973 | 35,308 | 41 |

These statistics describe the dataset reported in the paper. The current repository provides annotation CSVs, split lists, metadata, and reference code. Source broadcast videos are not included in this release.

## From Actions to Outcomes

JAS-2v2 uses a complete rally, from serve to point completion, as the sequence-level unit. Action annotations connect physical motion, action semantics, player positions, and rally outcomes.

<p align="center">
  <img src="assets/figures/action_to_score.png" alt="Comparison of match-level statistical analysis and action-level sequence modeling" width="1000">
</p>

*Figure 1 from the paper: moving from match-level statistics to action-level sequence modeling.*

The benchmark includes two tasks:

1. **Action category prediction:** predict an action category using the preceding action context.
2. **Score outcome prediction:** classify whether the serving team wins or loses the rally. The complete-rally setting is retrospective outcome classification; the paper also evaluates partially observed rallies.

Beach volleyball includes multi-contact teammate exchanges. The three racket sports involve direct exchanges across opposing sides. The dataset preserves these sport-specific action definitions while providing a common sequence-modeling setting.

## Dataset Statistics

The paper reports the following match-level train/test allocation:

| Sport | Matches, train/test | Sequences, train/test | Actions, train/test | Classes |
|---|---:|---:|---:|---:|
| Beach volleyball | 9 / 3 | 1,413 / 471 | 4,656 / 1,552 | 5 |
| Badminton doubles | 11 / 4 | 628 / 229 | 15,478 / 5,629 | 14 |
| Tennis doubles | 6 / 2 | 804 / 268 | 3,099 / 1,033 | 10 |
| Table tennis doubles | 9 / 3 | 870 / 290 | 2,896 / 965 | 12 |
| **Total** | **35 / 12** | **3,715 / 1,258** | **26,129 / 9,179** | **41** |

Annotations include action boundaries, action categories, acting-player court positions, and binary rally outcomes. Sport-specific class definitions and annotation guidelines are described in the paper.

## Reference Framework

The paper introduces a joint action-to-score framework with three modules:

- **Action Relation Learning (ARL):** learns predictive relations among action steps from their physical features.
- **Graph Construction:** combines physical descriptors and semantic cues in an action-graph representation.
- **Joint Prediction:** predicts action categories and uses the inferred action representation for sequence-level outcome prediction.

<p align="center">
  <img src="assets/figures/framework.png" alt="JAS-2v2 framework with action relation learning, graph construction, and joint prediction" width="1100">
</p>

*Figure 3 from the paper: the unified action-to-score framework. Learned relations are interpreted as predictive dependencies.*

The four physical descriptors used in the paper are:

| Descriptor | Meaning |
|---|---|
| **BV — Ball Velocity** | Average ball speed within an action segment |
| **PV — Player Velocity** | Average motion speed of the acting player within an action segment |
| **TD — Target Distance** | Distance between the estimated ball landing point and the position of the subsequent action executor |
| **PD — Player Distance** | Start-to-end displacement of the acting player within an action segment |

TD uses information about the subsequent action executor. Its availability must therefore be considered explicitly when evaluating online or partial-observation prediction. Reconstructing the physical descriptors requires aligned source footage or processed feature files.

## Repository Structure

```text
.
├── README.md
├── LICENSE.txt
├── metadata.json
├── requirements.txt
├── assets/
│   └── figures/                    # Figures extracted from the paper
├── code/
│   ├── dataset.py                  # CSV loading and sequence construction
│   ├── models.py                   # Lightweight graph-model components
│   ├── train_multitask_gcn.py       # Lightweight multi-task training entry point
│   ├── train_action_to_score.py     # Joint action-to-score reference implementation
│   └── extract_action_features.py  # Features from aligned footage and annotations
├── data/
│   ├── badminton/
│   ├── beachvolleyball/
│   ├── tennis/
│   └── table_tennis/
└── splits/
    ├── train_files.txt
    └── test_files.txt
```

## Getting Started

### 1. Download and install dependencies

```bash
git clone https://github.com/AngeIine/JAS_2v2.git
cd JAS_2v2
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your hardware when using CUDA.

### 2. Explore the annotations

CSV files are organized by sport under [`data/`](data). The release retains sport-specific column conventions:

| Content | Field names |
|---|---|
| Action category | `action_name` or `name` |
| Acting-player position | `position` |
| Binary outcome | `score` |
| Action boundaries | `start_frame` / `end_frame`, or timestamp-style `start` / `end` |

Processed feature files may also contain `target_distance`, `pdistance`, `pspeed`, and `bspeed`. Check the available columns before selecting a feature configuration. Timestamp-based annotations need frame alignment when reconstructing video-derived features.

The release split lists are in [`splits/`](splits). Match-level splitting is used in the paper. Check the split logic of the selected training script when reproducing a particular experiment.

### 3. Run the lightweight training scaffold

Run from the repository root:

```bash
python code/train_multitask_gcn.py --data-root data --split-mode mixed --feature-tier common
```

This is a lightweight entry point for the released annotation pipeline. Its defaults are separate from the full paper experiment configuration, and its split is constructed by the script rather than read automatically from `splits/`.

Inspect available settings with:

```bash
python code/train_multitask_gcn.py --help
python code/train_action_to_score.py --help
```

The joint action-to-score implementation expects suitable processed physical-feature CSVs. For feature extraction, see [`code/extract_action_features.py`](code/extract_action_features.py); aligned source videos and appropriate detection/pose models are required.

## Paper-Reported Results

Selected results on the unified four-sport benchmark, in percent:

| Method | Action Accuracy | Action F1 | Score Accuracy | Score F1 |
|---|---:|---:|---:|---:|
| Graphormer | 72.64 | 71.02 | 85.20 | 83.61 |
| Mamba | 73.35 | 71.90 | 86.42 | 84.85 |
| **Ours** | **74.28** | **72.91** | **87.45** | **86.13** |

These values are reported in the paper's Table 5 under its full-feature evaluation setting. The paper also includes feature/module ablations, temporal permutation tests, partial-rally evaluation, and leave-one-sport-out experiments.

## Citation

If you use JAS-2v2 in your research, please cite:

```bibtex
@inproceedings{wang2026beyondscore,
  title     = {Beyond Score: A Dataset for Joint Action and Score Predictions in 2v2 Sports},
  author    = {Wang, Yuting and Wen, Peng and Wang, Fengshun and Zuo, Jiayong and Peng, Xinyu and Wang, Qiurui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Usage Terms

Please read [`LICENSE.txt`](LICENSE.txt) for the release notice and usage terms. The annotation package is intended for non-commercial research and evaluation under the existing notice. Original broadcast footage remains subject to its owners' rights and applicable platform terms.

## Contact

- **Yuting Wang:** [wangyuting2023@cupes.edu.cn](mailto:wangyuting2023@cupes.edu.cn)
- **Qiurui Wang (corresponding author):** [wangqiurui@cupes.edu.cn](mailto:wangqiurui@cupes.edu.cn)

For questions about annotations or code, please [open an issue](https://github.com/AngeIine/JAS_2v2/issues).
