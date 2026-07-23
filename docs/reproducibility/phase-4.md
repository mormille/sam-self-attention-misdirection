# Phase 4 — Historical environment recovery

## Status

A minimal CPU execution path has been prepared for the archived Self-Attention Misdirection prototype. This is a compatibility reconstruction of the historical code, not validation of the final paper formulation or its performance.

## Compatibility environment

The GitHub Actions smoke test uses:

- Ubuntu 22.04 CPU runner
- Python 3.8
- PyTorch 1.8.1 CPU
- torchvision 0.9.1 CPU
- FastAI 2.3.1
- FastCore 1.3.20
- spaCy 2.3.1
- NumPy 1.20.3
- SciPy 1.6.3

The complete reconstructed dependency set is stored in `environment/legacy-requirements.txt`. The original repository did not include a complete environment lockfile.

## Smoke-test scope

Run:

```bash
python scripts/smoke_test.py
```

The smoke test follows the archived `launch-SAM.py` configuration by constructing the U-Net with `bilinear=False`. This is required because the historical truncated decoder is configured for transposed-convolution channel dimensions.

The test exercises:

1. residual image perturbation through the archived U-Net;
2. the archived ARViT critic on perturbed inputs;
3. reconstruction loss;
4. historical attention-misdirection loss;
5. classification/task loss;
6. the historical combined generator objective;
7. one generator optimizer step;
8. one critic optimizer step.

## Installation

```bash
python -m pip install pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install -r environment/legacy-requirements.txt
```

## Historical launch assumptions

The original launch path additionally assumes:

- a custom ImageNet directory under `Path.home() / 'Luiz/...'`;
- FastAI GAN internals modified by `models/utils/fastai_gan.py`;
- FP16 training;
- long multi-epoch training;
- serialized output directories that are not present;
- a six-layer, 512-dimensional, eight-head critic rather than the reduced smoke configuration.

The Phase 4 harness tests the underlying historical generator, critic, and losses directly rather than claiming recovery of the complete FastAI training loop.

## Checkpoint status

No current checkpoint binaries are present in the migrated SAM tree. Several historical checkpoint pointers were removed before the frozen source commit, and their corresponding Git LFS objects have not been recovered. Checkpoint loading therefore cannot be tested.

## Scope limitations

This phase does not establish:

- reproduction of a full SAM training run;
- agreement with the final two-page paper’s `MSE(S, S*)` formulation;
- validated performance improvement;
- recovery of the original custom ImageNet dataset;
- exact CUDA/FP16 behavior;
- licensing resolution for the historical U-Net implementation.

No archived model, loss, or training utility was modified.
