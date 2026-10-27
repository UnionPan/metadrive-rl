import torch

from metadrive_ais.models.latent_policy import LatentChannel


def test_prior_log_prob_is_exact_and_reproducible():
    torch.manual_seed(0)
    ch = LatentChannel(state_dim=16, latent_dim=3, token_dim=8)
    z = torch.randn(5, 16)
    dist = ch.prior(z)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    c1 = dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=g1)
    c2 = dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=g2)
    assert torch.equal(c1, c2)
    manual = (
        -((c1 - dist.mean) ** 2) / (2 * dist.stddev**2)
        - dist.stddev.log()
        - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    ).sum(-1)
    assert torch.allclose(dist.log_prob(c1).sum(-1), manual, atol=1e-5)


def test_posterior_uses_the_future_and_kl_is_nonnegative():
    torch.manual_seed(0)
    ch = LatentChannel(state_dim=16, latent_dim=3, token_dim=8)
    z, fut = torch.randn(4, 2, 16), torch.randn(4, 2, 6, 2) * 10
    post, post2 = ch.posterior(z, fut), ch.posterior(z, fut + 5.0)
    assert not torch.allclose(post.mean, post2.mean)
    kl = torch.distributions.kl_divergence(post, ch.prior(z)).sum(-1)
    assert kl.shape == (4, 2) and (kl >= 0).all()
    assert ch.decoder(post.rsample()).shape == (4, 2, 8)


def test_pca_encoder_gives_standardized_components_and_ignores_z():
    torch.manual_seed(0)
    t = torch.arange(1, 7, dtype=torch.float32)
    speed, curve = torch.rand(500, 1) * 15, torch.randn(500, 1) * 0.5
    futures = torch.stack([speed * 0.5 * t, curve * t**2], -1)  # [N, 6, 2]
    ch = LatentChannel(state_dim=16, latent_dim=2, token_dim=8, encoder="pca")
    explained = ch.fit_pca(futures)
    assert explained.sum() > 0.99  # two factors generate the data
    c = ch.posterior(torch.randn(500, 16), futures).mean
    assert torch.allclose(c.mean(0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(c.std(0), torch.ones(2), atol=1e-3)
    other_z = ch.posterior(torch.randn(500, 16), futures).mean
    assert torch.equal(c, other_z)
    assert ch.config["encoder"] == "pca"
    assert "encoder" not in LatentChannel(16, 2, 8).config  # loads older bundles


def test_fixed_rl_channel_is_frozen_orthonormal_and_starts_at_standard_normal():
    torch.manual_seed(0)
    ch = LatentChannel(
        state_dim=16, latent_dim=3, token_dim=8, decoder_scale=50.0, decoder_fixed=True
    )
    w = ch.decoder.proj.weight  # [token_dim, latent_dim]
    assert not w.requires_grad
    assert torch.allclose(w.T @ w, torch.eye(3), atol=1e-5)
    dist = ch.prior(torch.randn(5, 16))
    assert torch.equal(dist.mean, torch.zeros(5, 3)) and torch.allclose(
        dist.stddev, torch.ones(5, 3)
    )
    c = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.isclose(ch.decoder(c).norm(), torch.tensor(50.0))
    assert ch.config["decoder_fixed"] and ch.config["decoder_scale"] == 50.0
