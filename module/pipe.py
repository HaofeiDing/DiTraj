import torch
from diffusers import WanPipeline
from diffusers.utils import replace_example_docstring, is_torch_xla_available
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.wan import pipeline_wan
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

class myWanPipeline(WanPipeline):

    def create_self_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        构造分组自注意力掩码，使得前景只关注前景，背景只关注背景。
        
        参数:
            mask: 一维掩码，1表示前景，0表示背景。
        
        返回:
            torch.Tensor: 二维注意力掩码，形状为 [seq_len, seq_len]。
        """
        
        mask = mask.flatten()
        # 创建前景和背景的掩码矩阵
        # 前景元素可以关注其他前景元素
        foreground_mask = (mask == 1).unsqueeze(0) & (mask == 1).unsqueeze(1)
        # foreground_mask = foreground_mask & block_diag_mask
        # 背景元素可以关注其他背景元素
        background_mask = (mask == 0).unsqueeze(0) & (mask == 0).unsqueeze(1)
        # 合并前景和背景掩码
        attention_mask = foreground_mask | background_mask
        
        return attention_mask
    
    def create_cross_attention_mask(self, x_mask: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        """
        构造跨注意力掩码，使得前景只关注前景，背景只关注背景。
        
        参数:
            x_mask: 序列x的一维掩码，1表示前景，0表示背景。
            y_mask: 序列y的一维掩码，1表示前景，0表示背景。
        
        返回:
            torch.Tensor: 二维跨注意力掩码，形状为 [x_len, y_len]。
        """
        x_mask = x_mask.flatten()
        y_mask = y_mask.float()
        # 创建前景跨注意力：x的前景只能关注y的前景
        foreground_mask = (x_mask == 1).unsqueeze(1) & (y_mask == 1).unsqueeze(0)
        # 创建背景跨注意力：x的背景只能关注y的背景
        background_mask = (x_mask == 0).unsqueeze(1) & (y_mask == 0).unsqueeze(0)
        # 合并前景和背景掩码
        attention_mask = foreground_mask | background_mask
        
        return attention_mask

    def scale_complex_tensor(self, tensor, target_size):
        """
        缩放复数类型张量，使用最近邻插值方法
        
        参数:
            tensor: 输入复数张量，形状应为 (c, height, width)
            target_size: 目标尺寸，元组 (new_height, new_width)
        """
        # 获取原始尺寸和目标尺寸
        c, h, w = tensor.shape
        new_h, new_w = target_size
    
        
        # 生成目标坐标网格
        y = torch.linspace(0, h - 1, new_h, dtype=torch.float32).round().long()
        x = torch.linspace(0, w - 1, new_w, dtype=torch.float32).round().long()
        
        scaled_tensor = tensor[:, y[:,None], x]
        
        return scaled_tensor
    
    def create_fixrope_self_attention_mask(self, size, bbox_mask: torch.Tensor) -> torch.Tensor:
        f = size[0]
        h = size[1]//2
        w = size[2]//2
        i = torch.arange(f)[:, None, None]  # 形状: (f, 1, 1)
        j = torch.arange(h)[None, :, None]  # 形状: (1, h, 1)
        k = torch.arange(w)[None, None, :]  # 形状: (1, 1, w)
        
        # 扩展为相同形状并拼接，得到形状为(f, h, w, 3)的张量
        rope = torch.stack([
            i.expand(f, h, w),
            j.expand(f, h, w),
            k.expand(f, h, w)
        ], dim=-1)

        def find_corners(matrix):
            rows = len(matrix)
            if rows == 0:
                return None, None
            
            cols = len(matrix[0])
            top_left = (float('inf'), float('inf'))
            bottom_right = (-1, -1)
            
            for i in range(rows):
                for j in range(cols):
                    if matrix[i][j] == 1:
                        # 更新左上角坐标（行和列都最小）
                        if i < top_left[0] or (i == top_left[0] and j < top_left[1]):
                            top_left = (i, j)
                        # 更新右下角坐标（行和列都最大）
                        if i > bottom_right[0] or (i == bottom_right[0] and j > bottom_right[1]):
                            bottom_right = (i, j)
            # 如果未找到1，返回None
            if top_left[0] == float('inf'):
                return None, None
            return top_left, bottom_right
        
        (f0_h1, f0_w1), (f0_h2, f0_w2) = find_corners(bbox_mask[0].squeeze(0))
        # 基准为最小那一帧
        for frame_idx in range(1, bbox_mask.shape[0]):
            (fi_h1, fi_w1), (fi_h2, fi_w2) = find_corners(bbox_mask[frame_idx].squeeze(0))
            if (fi_h2 - fi_h1)*(fi_w2 - fi_w1) < (f0_h2- f0_h1)*(f0_w2 - f0_w1):
                f0_h1, f0_w1, f0_h2, f0_w2 = fi_h1, fi_w1, fi_h2, fi_w2

        for frame_idx in range(0, bbox_mask.shape[0]):
            (fi_h1, fi_w1), (fi_h2, fi_w2) = find_corners(bbox_mask[frame_idx].squeeze(0))
            tmp = rope[frame_idx].clone()
            
            if f0_h2-f0_h1 == fi_h2-fi_h1 and f0_w2-f0_w1 == fi_w2-fi_w1:
                # static bbox
                rope[frame_idx, fi_h1:fi_h2, fi_w1:fi_w2] = tmp[f0_h1:f0_h2, f0_w1:f0_w2]
            else:
                # dynamic bbox
                tmp1 = tmp[f0_h1:f0_h2, f0_w1:f0_w2].permute(2, 0, 1)  # (3, h, w)
                tmp2 = self.scale_complex_tensor(tmp1, (fi_h2-fi_h1, fi_w2-fi_w1))  # (3, fi_h2-fi_h1, fi_w2-fi_w1)
                tmp3 = tmp2.permute(1, 2, 0)  # (fi_h2-fi_h1, fi_w2-fi_w1, 3)
                rope[frame_idx, fi_h1:fi_h2, fi_w1:fi_w2] = tmp3

        rope = rope.reshape(-1, rope.shape[-1])
        equality = (rope.unsqueeze(1) == rope.unsqueeze(0)).all(dim=-1)
        # 将对角线元素设为False（排除自身比较）
        equality = equality & (~torch.eye(rope.shape[0], dtype=torch.bool, device=rope.device))

        overlap_token = equality.any(dim=1).to('cuda')
        mask_token = bbox_mask.flatten().bool()

        redundant_token = overlap_token & (~(overlap_token & mask_token))

        result = (redundant_token.unsqueeze(0) & mask_token.unsqueeze(1)) | (redundant_token.unsqueeze(1) & mask_token.unsqueeze(0))
    
        

        return ~result

    @torch.no_grad()
    @replace_example_docstring(pipeline_wan.EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            pipeline_wan.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds_base, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds_base = prompt_embeds_base.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        ## bg and fg prompt
        bg_prompt_embeds, _ = self.encode_prompt(
            prompt=attention_kwargs.pop('bg_prompt'),
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        fg_prompt_embeds, _ = self.encode_prompt(
            prompt=attention_kwargs.pop('fg_prompt'),
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        prompt_embeds_bgfg = torch.cat([bg_prompt_embeds, fg_prompt_embeds], dim=1).to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        bbox_mask = attention_kwargs['bbox_mask']
        fixRope_step = attention_kwargs['fixRope_step']
        mask_step = attention_kwargs['mask_step']
        # self_attention_mask = self.create_self_attention_mask(bbox_mask)
        self_attention_mask = None
        cross_attention_mask = self.create_cross_attention_mask(bbox_mask, attention_kwargs['encoder_attention_mask'].flatten())

        fixrope_self_attention_mask = self.create_fixrope_self_attention_mask(latents.shape[-3:], bbox_mask)
        #fixrope_self_attention_mask = None
        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                attention_kwargs = {}
                if i >= mask_step:
                    attention_kwargs['is_mask_step'] = False
                    prompt_embeds_now = prompt_embeds_base
                else:
                    attention_kwargs['is_mask_step'] = True
                    attention_kwargs['self_attention_mask'] = self_attention_mask
                    attention_kwargs['cross_attention_mask'] = cross_attention_mask
                    prompt_embeds_now = prompt_embeds_bgfg

                if i >= fixRope_step:
                    attention_kwargs['is_fixRope_step'] = False
                else:
                    attention_kwargs['is_fixRope_step'] = True
                    attention_kwargs['bbox_mask'] = bbox_mask
                    attention_kwargs['self_attention_mask'] = fixrope_self_attention_mask


                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_now,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    if i < mask_step:
                        attention_kwargs['cross_attention_mask'] = None
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if pipeline_wan.XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)