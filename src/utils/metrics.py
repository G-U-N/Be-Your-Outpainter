import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim
import lpips
from diffusers.models import AutoencoderKL

from src.utils.fvd.fvd import get_fvd_logits, frechet_distance
from src.utils.fvd.download import load_i3d_pretrained
from tqdm import tqdm
from einops import repeat

# gv means generated video
# gt means groundtruth video


class Metric:

    def __init__(self, device, lpips_type="alex", vae_path=None):
        self.device = device
        self.lpips_metric = lpips.LPIPS(net=lpips_type).to(device)
        self.vae = (
            AutoencoderKL.from_pretrained(vae_path).to(device)
            if vae_path is not None
            else None
        )

    # value 0-1
    # nxcxhxw
    @torch.no_grad()
    def compute_mse(self, gv, gt):
        if gv.shape != gt.shape:
            raise ValueError("Input tensors must have the same shape.")

        mse = torch.mean((gv - gt) ** 2)
        return mse

    # value 0-1
    # nxcxhxw
    @torch.no_grad()
    def compute_psnr(self, gv, gt):
        if gv.shape != gt.shape:
            raise ValueError("Input tensors must have the same shape.")

        mse = self.compute_mse(gv, gt)
        psnr = -10 * torch.log10(mse)

        return psnr

    # value 0-1
    # nxcxhxw
    @torch.no_grad()
    def compute_ssim(self, gv, gt, ms=False):
        if gv.shape != gt.shape:
            raise ValueError("Input tensors must have the same shape.")

        ssim_val = (
            ssim(gv, gt, data_range=1, size_average=True)
            if not ms
            else ms_ssim(gv, gt, data_range=1, size_average=True)
        )
        return ssim_val

    # value [-1,1]
    # nxcxhxw
    @torch.no_grad()
    def compute_lpips(self, gv, gt):
        gv = gv * 2 - 1
        gt = gt * 2 - 1
        lpips = self.lpips_metric.forward(gv, gt)
        return lpips

    @torch.no_grad()
    def compute_fvd(self, gvs, gts):
        i3d = load_i3d_pretrained(self.device)
        real_embeddings = []
        fake_embeddings = []

        for gv, gt in tqdm(zip(gvs, gts)):
            gv = gv.unsqueeze(0) * 255
            gt = gt.unsqueeze(0) * 255
            gv = gv.type(torch.uint8)
            gt = gt.type(torch.uint8)

            real_embeddings.append(
                get_fvd_logits(gv.cpu().numpy(), i3d=i3d, device=self.device)
            )
            fake_embeddings.append(
                get_fvd_logits(gt.cpu().numpy(), i3d=i3d, device=self.device)
            )

        real_embeddings = torch.cat(real_embeddings, dim=0)
        fake_embeddings = torch.cat(fake_embeddings, dim=0)
        fvd = frechet_distance(
            fake_embeddings.clone().detach(), real_embeddings.clone().detach()
        )

        return fvd


if __name__ == "__main__":
    metric_computer = Metric("cuda")

    gv = torch.rand((16, 3, 512, 512)).to("cuda")
    gt = gv

    mse = metric_computer.compute_mse(gv, gt)
    print(mse)
    psnr = metric_computer.compute_psnr(gv, gt)
    print(psnr)
    ssim_v = metric_computer.compute_ssim(gv, gt)
    print(ssim_v)
    lpips_v = metric_computer.compute_lpips(gv, gt)
    print(lpips)

    gv = torch.rand((2, 16, 3, 512, 512)).to("cuda")
    gt = gv
    fvd = metric_computer.compute_fvd(
        gv.permute(0, 1, 3, 4, 2), gt.permute(0, 1, 3, 4, 2)
    )
    print(fvd)
