"""
test_models.py — Unit tests for the .aFix neural network architecture.

These tests do NOT require trained weights, GPU, or internet access.
They verify architecture shapes and loss function behaviour using random tensors.

Run with:
    python -m pytest test_models.py -v
"""

import torch
import pytest

from model              import AfixVAE, AfixEncoder, AfixDecoder
from saliency_model     import SaliencyNet
from segmentation_model import SegmentationNet, OBJM_CATEGORIES
from loss               import AfixVAELoss, kl_divergence, saliency_l1


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vae():
    return AfixVAE()

@pytest.fixture
def sal_net():
    return SaliencyNet()

@pytest.fixture
def seg_net():
    return SegmentationNet()

@pytest.fixture
def small_img():
    """4× down-scaled tile for fast testing (128×128 instead of 512×512)."""
    return torch.rand(2, 3, 128, 128)


# ── VAE ───────────────────────────────────────────────────────────────────────

class TestAfixVAE:
    def test_encoder_output_shape(self, small_img):
        enc = AfixEncoder()
        mean, log_var = enc(small_img)
        B, C, H, W = small_img.shape
        # Latent spatial resolution = H/4 × W/4, 4 channels.
        assert mean.shape    == (B, 4, H // 4, W // 4)
        assert log_var.shape == (B, 4, H // 4, W // 4)

    def test_decoder_output_shape(self, small_img):
        enc = AfixEncoder()
        dec = AfixDecoder()
        mean, _ = enc(small_img)
        recon = dec(mean)
        assert recon.shape == small_img.shape, (
            f"Decoder output {recon.shape} should match input {small_img.shape}"
        )

    def test_vae_full_forward(self, vae, small_img):
        recon, mean, log_var = vae(small_img)
        assert recon.shape == small_img.shape
        assert mean.shape    == log_var.shape
        assert mean.shape[1] == vae.latent_channels

    def test_decoder_output_in_unit_range(self, vae, small_img):
        recon, _, _ = vae(small_img)
        assert recon.min() >= 0.0, "decoder output should be ≥ 0"
        assert recon.max() <= 1.0, "decoder output should be ≤ 1"

    def test_reparameterise_adds_noise_during_training(self):
        mean    = torch.zeros(1, 4, 8, 8, requires_grad=True)
        log_var = torch.zeros(1, 4, 8, 8)
        z1 = AfixEncoder.reparameterise(mean, log_var)
        z2 = AfixEncoder.reparameterise(mean, log_var)
        # Two samples should differ (with overwhelming probability).
        assert not torch.allclose(z1, z2), "reparameterisation should sample noise"

    def test_reparameterise_deterministic_at_inference(self):
        mean = torch.zeros(1, 4, 8, 8)  # requires_grad=False → inference mode
        log_var = torch.zeros(1, 4, 8, 8)
        z = AfixEncoder.reparameterise(mean, log_var)
        assert torch.allclose(z, mean), "inference should return mean"

    def test_encode_returns_mean(self, vae, small_img):
        z = vae.encode(small_img)
        assert z.shape[1] == vae.latent_channels

    def test_model_parameter_count(self, vae):
        params = sum(p.numel() for p in vae.parameters()) / 1e6
        # Should be < 20 M for a lightweight network.
        assert params < 20.0, f"VAE has too many parameters: {params:.1f} M"


# ── Saliency net ──────────────────────────────────────────────────────────────

class TestSaliencyNet:
    def test_output_shape(self, sal_net, small_img):
        sal = sal_net(small_img)
        B, _, H, W = small_img.shape
        assert sal.shape == (B, 1, H, W), f"Expected (B,1,H,W), got {sal.shape}"

    def test_output_in_unit_range(self, sal_net, small_img):
        sal = sal_net(small_img)
        assert sal.min() >= 0.0, "saliency should be ≥ 0"
        assert sal.max() <= 1.0, "saliency should be ≤ 1"

    def test_output_is_spatial(self, sal_net):
        # Different input → different saliency (not all zeros/ones).
        uniform = torch.full((1, 3, 64, 64), 0.5)
        noise   = torch.rand(1, 3, 64, 64)
        sal_u = sal_net(uniform)
        sal_n = sal_net(noise)
        assert not torch.allclose(sal_u, sal_n), "different inputs should give different saliency"

    def test_parameter_count(self, sal_net):
        params = sum(p.numel() for p in sal_net.parameters()) / 1e6
        assert params < 5.0, f"SaliencyNet has too many params: {params:.1f} M"


# ── Segmentation net ──────────────────────────────────────────────────────────

class TestSegmentationNet:
    def test_output_shape(self, seg_net, small_img):
        logits = seg_net(small_img)
        B, _, H, W = small_img.shape
        assert logits.shape == (B, len(OBJM_CATEGORIES), H, W)

    def test_mask_shape(self, seg_net, small_img):
        mask = seg_net.predict_mask(small_img)
        B, _, H, W = small_img.shape
        assert mask.shape == (B, H, W)

    def test_mask_values_in_class_range(self, seg_net, small_img):
        mask = seg_net.predict_mask(small_img)
        assert mask.min() >= 0
        assert mask.max() < len(OBJM_CATEGORIES)

    def test_predict_objects_returns_list(self, seg_net):
        img = torch.rand(1, 3, 64, 64)
        objects = seg_net.predict_objects(img)
        assert isinstance(objects, list)
        for obj in objects:
            assert "id"         in obj
            assert "label"      in obj
            assert "category"   in obj
            assert "bbox"       in obj
            assert "confidence" in obj
            conf = obj["confidence"]
            assert 0.0 <= conf <= 1.0, f"confidence {conf} out of [0,1]"

    def test_categories_match_spec(self):
        assert "background" in OBJM_CATEGORIES
        assert "subject"    in OBJM_CATEGORIES
        assert "sky"        in OBJM_CATEGORIES
        assert "ground"     in OBJM_CATEGORIES


# ── Loss function ─────────────────────────────────────────────────────────────

class TestAfixVAELoss:
    def test_all_loss_keys_present(self, vae, small_img):
        loss_fn = AfixVAELoss()
        recon, mean, log_var = vae(small_img)
        losses = loss_fn(recon, small_img, mean, log_var)
        for key in ("total", "structural", "neural", "pixel", "kl"):
            assert key in losses, f"Missing loss key: {key}"

    def test_total_is_positive(self, vae, small_img):
        loss_fn = AfixVAELoss()
        recon, mean, log_var = vae(small_img)
        losses = loss_fn(recon, small_img, mean, log_var)
        assert losses["total"].item() > 0.0

    def test_loss_with_saliency(self, vae, sal_net, small_img):
        loss_fn = AfixVAELoss(use_saliency=True)
        with torch.no_grad():
            saliency = sal_net(small_img)
        recon, mean, log_var = vae(small_img)
        losses = loss_fn(recon, small_img, mean, log_var, saliency)
        assert losses["total"].item() > 0.0

    def test_kl_zero_for_standard_normal(self):
        mean    = torch.zeros(2, 4, 8, 8)
        log_var = torch.zeros(2, 4, 8, 8)  # log(1) = 0 → variance = 1
        kl = kl_divergence(mean, log_var)
        assert abs(kl.item()) < 1e-4, f"KL of N(0,I) vs N(0,I) should be 0, got {kl.item()}"

    def test_saliency_l1_higher_for_mismatch(self):
        recon  = torch.zeros(1, 3, 8, 8)
        target = torch.ones(1, 3, 8, 8)
        # High saliency everywhere.
        sal_hi = torch.ones(1, 1, 8, 8)
        # Zero saliency everywhere.
        sal_lo = torch.zeros(1, 1, 8, 8)
        loss_hi = saliency_l1(recon, target, sal_hi)
        loss_lo = saliency_l1(recon, target, sal_lo)
        assert loss_hi > loss_lo, "high saliency should incur higher loss"

    def test_identical_recon_has_low_pixel_loss(self, vae, small_img):
        loss_fn = AfixVAELoss()
        target = small_img.clone()
        mean    = torch.zeros(2, 4, small_img.shape[2] // 4, small_img.shape[3] // 4)
        log_var = torch.zeros_like(mean)
        losses = loss_fn(target, target, mean, log_var)
        assert losses["pixel"].item() < 1e-5, "pixel loss for identical images should be ~0"
