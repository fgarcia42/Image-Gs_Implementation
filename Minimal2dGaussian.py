import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_coord_grid(H, W, device):
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing = "xy")
    grid = torch.stack([X, Y], dim =-1).permute(1, 0, 2).contiguous()
    return grid

def gaussian_exponent(grid, mu, log_scales, theta):
    H, W, _ = grid.shape
    K = mu.shape[0]

    scales = torch.exp(log_scales)
    sx = scales[:, 0]
    sy = scales[:, 1]

    diff = grid[:, :, None, :] - mu[None, None, :, :]
    dx = diff[..., 0]
    dy = diff[..., 1]

    c = torch.cos(theta)
    s = torch.sin(theta)
    vx = dx * c[None, None, :] + dy * s[None, None, :]
    vy = -dx * s[None, None, :] + dy * c[None, None, :]

    eps = 1e-6
    q = (vx * vx) / (sx[None, None, :] ** 2 + eps) + (vy * vy) / (sy[None, None, :] ** 2 + eps)
    return q

class GaussianImageModel(nn.Module):
    def __init__(self, H, W, K = 256, init_scale = 0.3, device='cpu'):
        super().__init__()
        self.H, self.W, self.K = H, W, K
        self.device = device

        self.mu = nn.Parameter(torch.empty(K, 2).uniform_(-0.8, 0.8))
        log_s = math.log(init_scale)
        self.log_scales = nn.Parameter(torch.full((K, 2), log_s))
        self.theta = nn.Parameter(torch.empty(K).uniform_(-math.pi, math.pi))
        self.color_logits = nn.Parameter(torch.empty(K, 3).normal_(0, 0.5))
        self.log_amp = nn.Parameter(torch.full((K, 1), math.log(0.5)))

        self.register_buffer("grid", make_coord_grid(H, W, device))

    def forward(self):
        q = gaussian_exponent(self.grid, self.mu, self.log_scales, self.theta)
        G = torch.exp(-0.5 * q)

        amp = F.softplus(self.log_amp).squeeze(-1)
        weights = G * amp[None, None, :]

        colors = torch.sigmoid(self.color_logits)

        num = torch.einsum("hwk,kc->hwc", weights, colors)
        den = weights.sum(dim=-1, keepdim=True)
        eps = 1e-6
        img = num / (den + eps)
        img = torch.clamp_(img, 0.0, 1.0)
        return img
    
    def loss(self, target, lambda_scale= 1e-4):
        pred = self()
        mse = F.mse_loss(pred, target)
        scale_reg = (self.log_scales**2).mean()
        p = mse + lambda_scale * scale_reg
        return p, pred

def fit_image_with_gaussians(target_img_np, K = 512, iters = 2000, lr=1e-2, device="cuda"):
    
    H, W, C = target_img_np.shape
    assert C == 3, "Target must be RGB"

    target = torch.from_numpy(target_img_np).to(device=device, dtype=torch.float32)
    model = GaussianImageModel(H, W, K =K, init_scale=0.25, device=device).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, iters + 1):
        opt.zero_grad()
        loss, pred = model.loss(target, lambda_scale = 1e-4)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.mu.clamp_(-1, 1)
            model.log_scales.clamp_(math.log(0.02), math.log(0.7))
        
        if step % 100 == 0 or step == 1:
            psnr = -10.0 * math.log10(float(F.mse_loss(pred, target)))
            print(f"[{step:04d} / {iters}] loss = {loss.item():.6f} psnr={psnr:.2f}db")
    
    with torch.no_grad():
        final = model().detach().cpu().numpy()
    return model, final

if __name__ == "__main__":
    try:
        from PIL import Image
        import matplotlib.pyplot as plt

        path = None #img path
        img = Image.open(path).convert("RGB")
        img = img.resize((128, 128), Image.LANCZOS)
        tgt = (np.asarray(img).astype(np.float32) / 255.0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, recon = fit_image_with_gaussians(tgt, K = 512, iters=1500, lr=5e-3, device=device)

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs[0].set_title("Target")
        axs[0].imshow(tgt)
        axs[0].axis("off")

        axs[1].set_title("Reconstruction")
        axs[1].imshow(recon)
        axs[1].axis("off")

        axs[2].set_title("Abs error")
        axs[2].imshow(np.abs(tgt - recon))
        axs[2].axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Demo visualization failed:", e)
