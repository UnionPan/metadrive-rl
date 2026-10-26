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
