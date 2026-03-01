import torch
import torch.nn as nn
from modules.ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from modules.ppd.models.dit import DiT
from modules.ppd.utils.sampler import EulerSampler
from modules.ppd.utils.schedule import LinearSchedule
from modules.ppd.utils.timesteps import Timesteps
from modules.ppd.utils.transform import image2tensor, resize_keep_aspect


class PixelPerfectDepth(nn.Module):
    def __init__(
        self,
        semantics_pth="checkpoints/depth_anything_v2_vitl.pth",
        sampling_steps=10,
        device: str = "cpu",
    ):
        super(PixelPerfectDepth, self).__init__()

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        self.device = torch.device(device)

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

        model_device = next(self.parameters()).device
        self.device = model_device
        image = image.to(model_device)

        use_mixed_precision = use_fp16 and model_device.type == "cuda"
        with torch.autocast(
            device_type=model_device.type, dtype=torch.float16, enabled=use_mixed_precision
        ):
            depth = self.forward_test(image)
        return depth, resize_image

    @torch.no_grad()
    def forward_test(self, image):
        semantics = self.semantics_prompt(image)
        cond = image - 0.5
        model_device = image.device
        latent = torch.randn(
            size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]],
            device=model_device,
        )

        for timestep in self.sampling_timesteps:
            timestep = timestep.to(model_device)
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)

        return latent + 0.5

    @torch.no_grad()
    def semantics_prompt(self, image):
        with torch.no_grad():
            semantics = self.semantics_encoder(image)
        return semantics
