---
license: other
license_name: nvidia-open-model-license
base_model: nvidia/Cosmos-Predict2-2B-Video2World
library_name: cosmos
pipeline_tag: image-to-video
tags:
  - robotics
  - world-model
  - action-conditioned-video
  - cosmos
---

# Prophet World Model

Built on NVIDIA Cosmos.

Prophet is the single-view, history-aware, action-conditioned robot world model introduced in **Reinforcing Action Policies by Prophesying**. It predicts future robot-observation videos from an initial observation, a 20-step action chunk, and an optional history buffer and projected action-frame sequence.

## Released Variants

- `prophet-pretrained.pt`: heterogeneous robot-data pretrained checkpoint used for downstream adaptation.
- `prophet-bridge.pt`: full-parameter BRIDGE-adapted checkpoint.
- `prophet-libero.pt`: full-parameter LIBERO-adapted checkpoint.

Exact sizes and SHA-256 checksums are listed in `weights.json`.

## Base Model and License

The checkpoints are derivative models of `nvidia/Cosmos-Predict2-2B-Video2World`. They are governed by the NVIDIA Open Model License included in `NVIDIA_OPEN_MODEL_LICENSE.md`. The repository source code is separately available under Apache-2.0.

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

## Intended Use

- Research on action-conditioned robot world models.
- Open-loop or autoregressive video rollout generation.
- Initialization for BRIDGE, LIBERO, or similarly formatted robot-data adaptation.
- Learned-simulator research with independent real-environment validation.

## Limitations

- Generated rollouts can hallucinate geometry, contacts, object state, or task success.
- Long autoregressive rollouts accumulate visual and dynamics errors.
- The model is not a replacement for real-robot safety validation or a certified simulator.
- Dataset-specific camera calibration and action semantics affect transfer quality.

## Citation

```bibtex
@article{zhang2025prophrl,
    title={Reinforcing Action Policies by Prophesying},
    author={Zhang, Jiahui and Huang, Ze and Gu, Chun and Ma, Zipei and Zhang, Li},
    year={2025},
    journal={arXiv preprint arXiv:2511.20633},
}
```
