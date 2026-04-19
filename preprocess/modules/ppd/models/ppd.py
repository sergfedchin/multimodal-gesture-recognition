import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from modules.ppd.models.dit import DiT
from modules.ppd.utils.sampler import EulerSampler
from modules.ppd.utils.schedule import LinearSchedule
from modules.ppd.utils.timesteps import Timesteps
from modules.ppd.utils.transform import image2tensor, pad_bgr_bottom_right, resize_keep_aspect


class PixelPerfectDepth(nn.Module):
    def __init__(
        self,
        semantics_pth="checkpoints/depth_anything_v2_vitl.pth",
        sampling_steps=10,
    ):
        super(PixelPerfectDepth, self).__init__()

        DEVICE = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.device = DEVICE

        self.semantics_encoder = DepthAnythingV2(
            encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
        )
        self.semantics_encoder.load_state_dict(
            torch.load(semantics_pth, map_location="cpu"), strict=False
        )
        self.semantics_encoder = self.semantics_encoder.to(self.device).eval()
        self.dit = DiT()

        self.sampling_steps = sampling_steps

        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.sampling_steps,
            device=self.device,
        )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type="velocity",
        )

    @torch.no_grad()
    def infer_image(self, image, use_fp16: bool = True):
        # Resize the image to match the training resolution area while keeping the original aspect ratio.
        resize_image = resize_keep_aspect(image)
        image = image2tensor(resize_image)
        image = image.to(self.device)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=use_fp16
        ):
            depth = self.forward_test(image)
        return depth, resize_image

    @torch.no_grad()
    def infer_images_batched(
        self,
        images_bgr,
        target_sizes_hw,
        use_fp16: bool = True,
        return_uint8: bool = True,
    ):
        """
        Batched depth inference with per-image resize_keep_aspect, pad to batch max H×W, one forward_test.

        Args:
            images_bgr: list of BGR uint8 arrays (e.g. person crops), same length as target_sizes_hw
            target_sizes_hw: list of (H, W) original RGB crop sizes to upsample depth maps to
            use_fp16: passed to autocast (same semantics as infer_image)
            return_uint8: if True, return uint8 min–max maps; if False, float32 HW tensors (CPU) before normalization

        Returns:
            list[np.ndarray] or list[torch.Tensor]: depth maps per image
        """
        if not images_bgr:
            return []
        if len(images_bgr) != len(target_sizes_hw):
            raise ValueError("images_bgr and target_sizes_hw must have the same length")

        resized_list = [resize_keep_aspect(im) for im in images_bgr]
        h_max = max(r.shape[0] for r in resized_list)
        w_max = max(r.shape[1] for r in resized_list)

        batch_parts = []
        for r in resized_list:
            padded = pad_bgr_bottom_right(r, h_max, w_max)
            batch_parts.append(image2tensor(padded))
        batch = torch.cat(batch_parts, dim=0).to(self.device)

        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=use_fp16
        ):
            depth = self.forward_test(batch)

        out_maps = []
        for i, (th, tw) in enumerate(target_sizes_hw):
            d = depth[i : i + 1]
            d = F.interpolate(d, size=(th, tw), mode="bilinear", align_corners=False)[
                0, 0
            ]
            if not return_uint8:
                out_maps.append(d.float().cpu())
                continue
            depth_np = d.float().cpu().numpy()
            depth_normalized = (depth_np - depth_np.min()) / (
                depth_np.max() - depth_np.min() + 1e-8
            )
            out_maps.append((depth_normalized * 255).astype(np.uint8))
        return out_maps


    @torch.no_grad()
    def forward_test(self, image):
        semantics = self.semantics_prompt(image)
        cond = image - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(
            self.device
        )

        for timestep in self.sampling_timesteps:
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)

        return latent + 0.5

    @torch.no_grad()
    def semantics_prompt(self, image):
        with torch.no_grad():
            semantics = self.semantics_encoder(image)
        return semantics


def verify_infer_images_batched_parity(
    model: "PixelPerfectDepth",
    image_bgr,
    target_hw,
    seed: int = 42,
    rtol: float = 0.05,
    atol: float = 0.05,
) -> None:
    """Assert B=1 batched API matches single forward + interpolate (fp16-tolerant)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resized = resize_keep_aspect(image_bgr)
    t_single = image2tensor(resized).to(model.device)
    with torch.autocast(
        device_type=model.device.type, dtype=torch.float16, enabled=True
    ):
        d_single = model.forward_test(t_single)
    d_single = F.interpolate(
        d_single,
        size=(target_hw[0], target_hw[1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].float().cpu()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batched = model.infer_images_batched(
        [image_bgr], [target_hw], use_fp16=True, return_uint8=False
    )
    assert len(batched) == 1
    torch.testing.assert_close(d_single, batched[0], rtol=rtol, atol=atol)

    uint_maps = model.infer_images_batched(
        [image_bgr], [target_hw], use_fp16=True, return_uint8=True
    )
    assert len(uint_maps) == 1 and uint_maps[0].shape == (target_hw[0], target_hw[1])
