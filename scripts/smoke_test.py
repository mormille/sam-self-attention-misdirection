#!/usr/bin/env python3
"""Minimal alternating-step smoke test for the archived SAM prototype."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.sam_loss import CriticLoss, GeneratorLoss, Misdirection_loss  # noqa: E402
from models.ARViT import ARViT  # noqa: E402
from models.unet import UNet  # noqa: E402


def main() -> None:
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    batch_size = 2

    # Match the historical launch-SAM.py configuration. The truncated decoder
    # is configured for transposed convolutions rather than bilinear upsampling.
    generator = UNet(n_channels=3, n_classes=3, bilinear=False)
    critic = ARViT(
        num_encoder_layers=1,
        nhead=4,
        num_classes=3,
        batch_size=batch_size,
        hidden_dim=32,
        image_h=256,
        image_w=256,
        grid_l=16,
        gm_patch=16,
    )

    images = torch.rand(batch_size, 3, 256, 256)
    labels = torch.tensor([0, 1], dtype=torch.long)

    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.SGD(
        [parameter for parameter in critic.parameters() if parameter.requires_grad],
        lr=1e-5,
    )

    # Historical generator step: reconstruction + attention misdirection - task loss.
    generator_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    generated_pair = generator(images)
    noised_images, original_images = generated_pair
    assert noised_images.shape == images.shape
    assert original_images.shape == images.shape

    generator_output = critic(noised_images)
    reconstruction = nn.MSELoss()(original_images, noised_images)
    misdirection = Misdirection_loss(bias=-0.17)(
        generator_output[1][0], generator_output[3]
    )
    classification = nn.CrossEntropyLoss()(generator_output[0], labels)
    generator_loss = GeneratorLoss(
        beta=0.001,
        bias=-0.17,
        layers=[0],
        gammas=[0.0002],
    )(generator_output, generated_pair, labels)

    for value, name in [
        (reconstruction, "reconstruction"),
        (misdirection, "misdirection"),
        (classification, "classification"),
        (generator_loss, "generator"),
    ]:
        if not torch.isfinite(value):
            raise RuntimeError(f"Non-finite {name} loss: {value}")

    generator_loss.backward()
    generator_optimizer.step()
    if not any(parameter.grad is not None for parameter in generator.parameters()):
        raise RuntimeError("Generator did not receive gradients")

    # Historical critic step on a detached generated image.
    critic_optimizer.zero_grad()
    with torch.no_grad():
        detached_noised = generator(images)[0]
    critic_output = critic(detached_noised.detach())
    critic_loss = CriticLoss(
        layers=[0],
        bias=-0.17,
        lambdas=[0.002],
    )(critic_output, labels)
    if not torch.isfinite(critic_loss):
        raise RuntimeError(f"Non-finite critic loss: {critic_loss}")
    critic_loss.backward()
    critic_optimizer.step()

    if not any(parameter.grad is not None for parameter in critic.parameters() if parameter.requires_grad):
        raise RuntimeError("Critic did not receive gradients")

    print("PASS: U-Net perturbation, three generator components, and alternating generator/critic steps")
    print(
        f"torch={torch.__version__}; reconstruction={float(reconstruction.detach()):.6f}; "
        f"misdirection={float(misdirection.detach()):.6f}; critic={float(critic_loss.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
