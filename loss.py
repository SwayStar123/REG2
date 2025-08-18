import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None, cls_token=None,
                 time_input=None, noises=None,):
        if model_kwargs == None:
            model_kwargs = {}
        # Support cls_token as a list of tensors or a tensor with optional token dimension.
        # Acceptable input forms:
        #  - Tensor shape (B, D)
        #  - Tensor shape (B, K, D)
        #  - List[Tensor] each of shape (B, D) -> stacked to (B, K, D)
        if isinstance(cls_token, list):
            # Validate consistent shapes then stack
            assert all(t.ndim == 2 for t in cls_token), "Each cls token tensor in the list must have shape (B, D)."
            cls_token = torch.stack(cls_token, dim=1)  # (B, K, D)
        elif isinstance(cls_token, torch.Tensor):
            if cls_token.ndim == 2:
                cls_token = cls_token.unsqueeze(1)  # (B, 1, D)
            elif cls_token.ndim == 3:
                pass  # already (B, K, D)
            else:
                raise ValueError("cls_token must have shape (B, D) or (B, K, D)")
        else:
            raise ValueError("cls_token must be a Tensor or list of Tensors")

        # sample timesteps
        if time_input is None:
            if self.weighting == "uniform":
                time_input = torch.rand((images.shape[0], 1, 1, 1))
            elif self.weighting == "lognormal":
                # sample timestep according to log-normal distribution of sigmas following EDM
                rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
                sigma = rnd_normal.exp()
                if self.path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)

        if noises is None:
            noises = torch.randn_like(images)
            noises_cls = torch.randn_like(cls_token)  # (B, K, D)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        # Reshape alphas to broadcast over cls tokens: alpha_t: (B,1,1,1) -> (B,1,1)
        alpha_scalar = alpha_t.view(alpha_t.shape[0], 1, 1)
        sigma_scalar = sigma_t.view(sigma_t.shape[0], 1, 1)

        model_input = alpha_t * images + sigma_t * noises
        cls_input = alpha_scalar * cls_token + sigma_scalar * noises_cls
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
            cls_target = d_alpha_t * cls_token + d_sigma_t * noises_cls
        else:
            raise NotImplementedError()

        model_output, zs_tilde, cls_output = model(
            model_input,
            time_input.flatten(),
            **model_kwargs,
            cls_token=cls_input
        )

        #denoising_loss
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        # cls_output may be (B, K, Dcls); we keep per-sample mean across K & D
        denoising_loss_cls = mean_flat((cls_output - cls_target) ** 2)


        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                # z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1, eps=1e-7) 
                # z_j = torch.nn.functional.normalize(z_j, dim=-1, eps=1e-7)
                # proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))

                # cosine with upcasting
                z_tilde_j_fp32 = z_tilde_j.float()
                z_j_fp32 = z_j.float()

                # Save first 5 values before normalization (flatten to 1D)
                # if i == 0 and j == 0:
                    # print("z_j pre-norm:", z_j_fp32.flatten()[:5].detach().cpu())
                    # print("z_tilde_j pre-norm:", z_tilde_j_fp32.flatten()[:5].detach().cpu())

                z_tilde_j_norm = torch.nn.functional.normalize(z_tilde_j_fp32, dim=-1)
                z_j_norm = torch.nn.functional.normalize(z_j_fp32, dim=-1)

                # Cast back to original dtype
                z_tilde_j_norm = z_tilde_j_norm.to(z_tilde_j.dtype)
                z_j_norm = z_j_norm.to(z_j.dtype)

                # Print first 5 values after normalization
                # if i == 0 and j == 0:
                #     print("z_j post-norm:", z_j_norm.flatten()[:5].detach().cpu())
                #     print("z_tilde_j post-norm:", z_tilde_j_norm.flatten()[:5].detach().cpu())

                proj_loss += mean_flat(-(z_j_norm * z_tilde_j_norm).sum(dim=-1))

                # mse
                # proj_loss += mean_flat((z_j - z_tilde_j).pow(2).sum(dim=-1)) * 0.1

                # capped mse
                # z_tilde_j = torch.clamp(z_tilde_j, -1.0, 1.0)
                # proj_loss += mean_flat((z_j - z_tilde_j).pow(2).sum(dim=-1))



        proj_loss /= (len(zs) * bsz)

        # For backward compatibility, if only one cls token supplied, downstream code can squeeze K dim.
        return denoising_loss, proj_loss, time_input, noises, denoising_loss_cls