import torch
import torch.nn as nn
import torch.distributions as dist

class CrossModalLosses(nn.Module):
    def __init__(self, lambda_kl=1e-5, lambda_e=1e-5, beta_kl=1.0):
        super(CrossModalLosses, self).__init__()
        # Define the SmoothL1 loss for cross-modal similarity
        self.smooth_l1_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=beta_kl)
        
        # Scaling factors for KL divergence and cross-modal similarity losses
        self.lambda_kl = lambda_kl
        self.lambda_e = lambda_e

    def kl_divergence_loss(self, mu_T, logvar_T, mu_M, logvar_M):
        """Compute the KL divergence loss for aligning text and motion embeddings."""
        
        # Convert log-variance to standard deviation (scale)
        sigma_T = torch.exp(0.5 * logvar_T)
        sigma_M = torch.exp(0.5 * logvar_M)
        
        # Define the two distributions
        dist_T = dist.Normal(mu_T, sigma_T)
        dist_M = dist.Normal(mu_M, sigma_M)

        # Standard Normal prior N(0, I)
        mu_ref = torch.zeros_like(mu_T)
        sigma_ref = torch.ones_like(sigma_T)
        dist_prior = dist.Normal(mu_ref, sigma_ref)

        # Compute KL divergences
        kl_T_M = torch.distributions.kl.kl_divergence(dist_T, dist_M).sum(dim=-1)
        kl_M_T = torch.distributions.kl.kl_divergence(dist_M, dist_T).sum(dim=-1)

        kl_T_prior = torch.distributions.kl.kl_divergence(dist_T, dist_prior).sum(dim=-1)
        kl_M_prior = torch.distributions.kl.kl_divergence(dist_M, dist_prior).sum(dim=-1)

        # Total KL loss
        L_KL = kl_T_M + kl_M_T + kl_T_prior + kl_M_prior

        return L_KL.mean()  # Take mean over batch
    
    def cross_modal_embedding_similarity_loss(self, z_t, z_m):
        """Compute the L1 similarity loss between text and motion embeddings."""
        return self.smooth_l1_loss_fn(z_t, z_m)

    def reconstruction_loss(self, H_gt, H_hat_M, H_hat_T):
        """Compute the reconstruction loss comparing ground truth to motion and text reconstructions."""
        # L1 loss between ground truth human motion (H1:F) and reconstructions (̂ HM 1:F, ̂ HT 1:F)
        loss_M = self.smooth_l1_loss_fn(H_gt, H_hat_M)
        loss_T = self.smooth_l1_loss_fn(H_gt, H_hat_T)
        return loss_M + loss_T
    
    def forward(self, mu_T, std_T, mu_M, std_M, z_t, z_m, H_gt, H_hat_M, H_hat_T):
        """Compute the total loss including KL divergence, cross-modal similarity, and reconstruction."""
        
        # Compute KL divergence loss
        kl_loss = self.kl_divergence_loss(mu_T, std_T, mu_M, std_M)
        
        # Compute cross-modal embedding similarity loss
        embedding_similarity_loss = self.cross_modal_embedding_similarity_loss(z_t, z_m)
        
        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_loss(H_gt, H_hat_M, H_hat_T)
        
        # Total loss = L_R + λ_KL * L_KL + λ_E * L_E
        total_loss = reconstruction_loss + self.lambda_kl * kl_loss + self.lambda_e * embedding_similarity_loss
        
        return total_loss
