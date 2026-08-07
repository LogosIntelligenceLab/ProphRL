# Third-Party Notices

The repository-level MIT license covers original VLA-RL code only. The
components below retain their own copyright notices and license terms. Source
headers remain authoritative when they are more specific than this summary.

## veRL

- Source: <https://github.com/volcengine/verl>
- License: Apache-2.0, see `LICENSES/Apache-2.0.txt`
- Copyright: 2024 ByteDance Ltd. and/or its affiliates
- Vendored scope: the distributed training framework under `verl_vla/`

`verl_vla` originated from veRL and contains local VLA-specific modifications.
The standalone upstream snapshot under `verl/` is not part of the intended
public package.

## vLLM

- Source: <https://github.com/vllm-project/vllm>
- License: Apache-2.0, see `LICENSES/Apache-2.0.txt`
- Copyright: 2023 The vLLM team; subsequent ByteDance modifications where noted
- Vendored scope: `verl_vla/third_party/vllm/` and files whose headers identify
  vLLM adaptations

The repository contains compatibility adapters for several vLLM API versions.
The directory names record the corresponding upstream version families.

## OpenPI and Big Vision

- Source: <https://github.com/Physical-Intelligence/openpi>
- License: Apache-2.0, see `LICENSES/Apache-2.0.txt`
- Vendored scope: `verl_vla/utils/vla_utils/pi05/openpi/`

Some Pi05 modules are adapted from Google's Big Vision project and retain the
Big Vision copyright headers. Use of Gemma model weights or model derivatives
is additionally governed by `LICENSES/Gemma-Terms-of-Use.txt`.

## OpenVLA-OFT

- Source: <https://github.com/moojink/openvla-oft>
- License: MIT, see `LICENSES/OpenVLA-OFT-MIT.txt`
- Vendored scope: `verl_vla/utils/vla_utils/openvla/`,
  `verl_vla/utils/vla_utils/openvla_oft/`, and
  `verl_vla/utils/vla_utils/openvla_oft_flow/`

## VLA-Adapter

- Source: <https://github.com/OpenHelix-Team/VLA-Adapter>
- License: MIT, see `LICENSES/VLA-Adapter-MIT.txt`
- Vendored scope: `verl_vla/utils/vla_utils/vla_adapter/`

## Shanghai Innovation Institute RL Environment Adapter

- License: Apache-2.0, see `LICENSES/Apache-2.0.txt`
- Copyright: 2025 Shanghai Innovation Institute
- Vendored scope: `verl_vla/environment/`

These files were extracted from the former siiRL snapshot and adapted to use
package-local imports, standard logging, and optional LIBERO imports.

## Hugging Face Transformers

- Source: <https://github.com/huggingface/transformers>
- License: Apache-2.0, see `LICENSES/Apache-2.0.txt`
- Vendored scope: files whose headers or source comments identify Transformers
  adaptations, including Pi05 replacement modules and Prismatic HF adapters

## Attribution Maintenance

Before a public release, every retained vendored component must record an exact
upstream tag or commit and a summary of local modifications. Removing a
component requires removing its stale notice only after a repository-wide
reference scan confirms that no derived code remains.
