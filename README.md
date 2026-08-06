<div align="center">

# Reinforcing Action Policies by *Prophesying*

[Paper](https://arxiv.org/abs/2511.20633) · [Project Page](https://LogosRoboticsGroup.github.io/ProphRL/) · [Models](https://huggingface.co/Fleurrr/Prophet-World-Model)

Jiahui Zhang<sup>1,3,*</sup>, Ze Huang<sup>1,3,*</sup>, Chun Gu<sup>1,2,3</sup>, Zipei Ma<sup>1,2</sup>, [Li Zhang](https://lzrobots.github.io/)<sup>1,2,3</sup>

<sup>1</sup>School of Data Science, Fudan University · <sup>2</sup>Shanghai Innovation Institute · <sup>3</sup>Logos Robotics · <sup>*</sup>Equal contribution

<img src="docs/resources/teaser.png" width="100%" alt="ProphRL overview"/>

</div>

## Overview

ProphRL improves vision-language-action policies through reinforcement learning inside a learned robot world model. Its world model, **Prophet**, predicts future robot observations from an initial frame, action chunks, and visual history. The full method additionally introduces FA-GRPO and FlowScale for VLA post-training.

## Abstract

Vision-Language-Action policies are commonly trained through imitation and can be brittle under distribution shift, while real-world reinforcement learning is expensive and conventional simulators are difficult to engineer and transfer. ProphRL addresses these limitations with Prophet, a unified action-to-video robot world model pretrained on heterogeneous robot trajectories and adapted to downstream environments through full-parameter fine-tuning. The complete method reinforces VLA policies inside Prophet using Flow-action-GRPO and FlowScale. This release focuses on the Prophet world-model component; the RL component will be released separately.

This repository is organized by method component so that the world-model implementation and the forthcoming RL implementation remain independent:

```text
ProphRL/
├── world_model/     # Prophet training, checkpoints, and rollout generation
├── rl/              # FA-GRPO and FlowScale; to be released
└── docs/            # Project assets
```

## Current Release

The current code release contains the Prophet world model:

- public pretrained, BRIDGE, and LIBERO checkpoints;
- BRIDGE RLDS and LIBERO HDF5 data conversion tools;
- full-parameter BRIDGE and LIBERO fine-tuning;
- an automatic-download inference demo plus open-loop/autoregressive rollout generation;
- checkpoint validation and release metadata.

See [`world_model/README.md`](world_model/README.md) for installation and usage.

## Quick Start

```bash
cd world_model
python -m pip install -e . --no-deps
python scripts/download_base_models.py

prophet-demo \
  --variant libero \
  --input /path/to/conditioning.mp4 \
  --actions /path/to/actions.npz \
  --base-checkpoints checkpoints \
  --output outputs/libero_demo.mp4
```

`prophet-demo` downloads the selected Prophet checkpoint from Hugging Face automatically. Data conversion and fine-tuning commands are documented in [`world_model/docs/DATA.md`](world_model/docs/DATA.md) and [`world_model/README.md`](world_model/README.md).

The RL implementation, reward-model code, and private real-robot integrations are not part of this release yet.

## License

The source code is distributed under Apache-2.0. Prophet checkpoints are derivative models of NVIDIA Cosmos-Predict2 and are distributed separately under the NVIDIA Open Model License. See `world_model/release/` for the model card, license, and checksums.

Built on NVIDIA Cosmos.

## Citation

```bibtex
@article{zhang2025prophrl,
    title={Reinforcing Action Policies by Prophesying},
    author={Zhang, Jiahui and Huang, Ze and Gu, Chun and Ma, Zipei and Zhang, Li},
    year={2025},
    journal={arXiv preprint arXiv:2511.20633},
}
```
