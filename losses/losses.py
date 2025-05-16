import torch
import torch.nn as nn
import torch.distributions as dist

class CrossModalLosses(nn.Module):
    def __init__(self, lambda_kl=1e-5, lambda_e=1e-5, beta_kl=1.0):
        super(CrossModalLosses, self).__init__()
        # Define the SmoothL1 loss for cross-modal similarity
        self.smooth_l1_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=beta_kl)
        self.smooth_l1_loss_fn_none = nn.SmoothL1Loss(reduction='none', beta=beta_kl)
        
        # Scaling factors for KL divergence and cross-modal similarity losses
        self.lambda_kl = lambda_kl
        self.lambda_e = lambda_e

    # def kl_divergence_loss(self, mu_T, sigma_T, mu_M, sigma_M):
    #     """Compute the KL divergence loss for aligning text and motion embeddings."""
        
    #     # Convert log-variance to standard deviation (scale)
    #     # sigma_T = torch.exp(0.5 * logvar_T)
    #     # sigma_M = torch.exp(0.5 * logvar_M)
        
    #     # Define the two distributions
    #     dist_T = dist.Normal(mu_T, sigma_T)
    #     dist_M = dist.Normal(mu_M, sigma_M)

    #     # Standard Normal prior N(0, I)
    #     mu_ref = torch.zeros_like(mu_T)
    #     sigma_ref = torch.ones_like(sigma_T)
    #     dist_prior = dist.Normal(mu_ref, sigma_ref)

    #     # Compute KL divergences
    #     kl_T_M = torch.distributions.kl.kl_divergence(dist_T, dist_M).sum(dim=-1)
    #     kl_M_T = torch.distributions.kl.kl_divergence(dist_M, dist_T).sum(dim=-1)

    #     kl_T_prior = torch.distributions.kl.kl_divergence(dist_T, dist_prior).sum(dim=-1)
    #     kl_M_prior = torch.distributions.kl.kl_divergence(dist_M, dist_prior).sum(dim=-1)

    #     # Total KL loss
    #     L_KL = kl_T_M + kl_M_T + kl_T_prior + kl_M_prior

    #     return L_KL.mean()  # Take mean over batch

    def kl_divergence_loss(self, dist_T, dist_M):
        """Compute KL divergence loss using distributions directly."""

        # Standard Normal prior N(0, I)
        dist_prior = dist.Normal(torch.zeros_like(dist_T.mean), torch.ones_like(dist_T.stddev))

        # Compute KL divergences
        kl_T_M = torch.distributions.kl.kl_divergence(dist_T, dist_M).mean()
        kl_M_T = torch.distributions.kl.kl_divergence(dist_M, dist_T).mean()
        kl_T_prior = torch.distributions.kl.kl_divergence(dist_T, dist_prior).mean()
        kl_M_prior = torch.distributions.kl.kl_divergence(dist_M, dist_prior).mean()

        # Total KL loss
        return (kl_T_M + kl_M_T + kl_T_prior + kl_M_prior)
    
    def cross_modal_embedding_similarity_loss(self, z_t, z_m):
        """Compute the L1 similarity loss between text and motion embeddings."""
        return self.smooth_l1_loss_fn(z_t, z_m)

    # def reconstruction_loss(self, H_gt, H_hat_T, H_hat_M):
    #     """Compute the reconstruction loss comparing ground truth to motion and text reconstructions."""
    #     # L1 loss between ground truth human motion (H1:F) and reconstructions (̂ HM 1:F, ̂ HT 1:F)
    #     loss_M = self.smooth_l1_loss_fn(H_gt, H_hat_M)
    #     loss_T = self.smooth_l1_loss_fn(H_gt, H_hat_T)
    #     print(f'{loss_M=}')
    #     print(f'{loss_T=}')
    #     return loss_M + loss_T

    def reconstruction_loss(self, H_gt, H_hat_M, H_hat_T, lengths):
        # print(lengths)
        lengths = torch.tensor(lengths).to('cuda')
        """Compute reconstruction loss while ignoring padded regions."""
        mask = torch.arange(H_gt.shape[1], device=H_gt.device).unsqueeze(0) < lengths.unsqueeze(1)  # (batch, time)
        mask = mask.unsqueeze(-1).expand_as(H_gt)  # (batch, time, 1) for broadcasting

        loss_M = self.smooth_l1_loss_fn(H_gt[mask], H_hat_M[mask])
        loss_T = self.smooth_l1_loss_fn(H_gt[mask], H_hat_T[mask])

        # print(f'{loss_M=}')
        # print(f'{loss_T=}')

        return loss_M + loss_T
    
    def reconstruction_loss1(self, H_gt, H_hat_M, H_hat_T, lengths):
        """
        Compute reconstruction loss using element-wise Smooth L1 loss,
        masking out the padded regions based on sequence lengths.

        Args:
            H_gt (Tensor): Ground truth tensor (batch_size, seq_len, dim)
            H_hat_M (Tensor): Predicted output from model M (same shape)
            H_hat_T (Tensor): Predicted output from model T (same shape)
            lengths (List[int] or Tensor): Lengths of valid sequences in the batch

        Returns:
            Tensor: Scalar loss value (sum of masked losses from M and T)
        """
        lengths = torch.as_tensor(lengths, device=H_gt.device)

        # Create a mask for valid time steps (batch_size, seq_len, 1)
        mask = torch.arange(H_gt.size(1), device=H_gt.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(H_gt)  # (batch_size, seq_len, dim)

        # Compute unreduced Smooth L1 loss
        loss_M = self.smooth_l1_loss_fn_none(H_hat_M, H_gt)
        loss_T = self.smooth_l1_loss_fn_none(H_hat_T, H_gt)

        # print(loss_M + loss_T)
        # print((loss_M + loss_T).shape)

        # Apply mask and compute mean loss over valid elements
        masked_loss_M = loss_M[mask].mean()
        masked_loss_T = loss_T[mask].mean()

        return masked_loss_M + masked_loss_T
    
    def reconstruction_loss2(self, H_gt, H_hat_M, H_hat_T, lengths):
        # H_gt, H_hat_M, H_hat_T: [B, T, F]
        B, T, F = H_gt.shape
        mask = torch.arange(T, device=H_gt.device).unsqueeze(0) < torch.tensor(lengths, device=H_gt.device).unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(-1, -1, F)  # [B, T, F]

        # Use reduction='none' so we control averaging
        loss_fn = nn.SmoothL1Loss(reduction='none', beta=self.smooth_l1_loss_fn.beta)
        loss_M = loss_fn(H_hat_M, H_gt)
        loss_T = loss_fn(H_hat_T, H_gt)

        # Apply mask
        loss_M = loss_M * mask
        loss_T = loss_T * mask

        # Normalize by number of valid elements
        num_valid = mask.sum()
        loss = (loss_M.sum() + loss_T.sum())# / num_valid.clamp(min=1)

        return loss
    
    def forward(self, dist_T, dist_M, z_t, z_m, H_gt, H_hat_T, H_hat_M, lengths):
        """Compute the total loss including KL divergence, cross-modal similarity, and reconstruction."""
        
        # Compute KL divergence loss
        kl_loss = self.kl_divergence_loss(dist_T, dist_M)
        # print(f'{kl_loss=}')

        
        # Compute cross-modal embedding similarity loss
        embedding_similarity_loss = self.cross_modal_embedding_similarity_loss(z_t, z_m)
        # print(f'{embedding_similarity_loss=}')

        
        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_loss2(H_gt, H_hat_T, H_hat_M, lengths)
        
        # Total loss = L_R + λ_KL * L_KL + λ_E * L_E
        total_loss = reconstruction_loss + self.lambda_kl * kl_loss + self.lambda_e * embedding_similarity_loss
        
        # print(f'{total_loss=}\n')

        return total_loss, kl_loss, embedding_similarity_loss, reconstruction_loss
