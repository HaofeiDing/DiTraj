import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from module.attention_processor import MyWanAttnProcessor2_0
from module.pipe import myWanPipeline
from utils import bboxs_to_arg, plan_path, arg_to_bboxs, save_videos_with_bbox
import random 
import numpy as np
import argparse
import json

torch.set_grad_enabled(False)

# ######################################Define your own trajectory#######################################
# Simple trajectory
bboxs = [
            [0, 0.3, 0.7, 0.1, 0.4], # frame 0: Left side
            [80, 0.3, 0.7, 0.7,1.0]  # frame 80: Right side
        ]
# Complex trajectory
# bboxs = [
#             [0, 0.05, 0.55, 0.05, 0.45], # frame 0: Top-left
#             [20, 0.05, 0.55, 0.55, 0.95], # frame 20: Top-right
#             [40, 0.45, 0.95, 0.55, 0.95], # frame 40: Bottom-left
#             [60, 0.45, 0.95, 0.05, 0.45], # frame 60: Bottom-right
#             [80, 0.05, 0.55, 0.05, 0.45], # frame 80: Top-left 
#         ]
bboxs_flat = [str(num) for bbox in bboxs for num in bbox]
bboxs_arg = ",".join(bboxs_flat) 
##########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=25234, type=int)
parser.add_argument("--mask_step", default=30, type=int)
parser.add_argument("--output_path", default='demo/output.mp4', type=str)
parser.add_argument("--output_path_withbox", default='demo/output_box.mp4', type=str)
parser.add_argument("--bboxs_arg", default=bboxs_arg, type=str)
parser.add_argument("--fixRope_step", default=5, type=int)
parser.add_argument("--num_frame", default=81, type=int)
parser.add_argument("--height", default=480, type=int)
parser.add_argument("--width", default=832, type=int)
parser.add_argument("--fps", default=4, type=int)
parser.add_argument("--prompt_json", default="demo/test_prompts_refined.json", type=str)

args = parser.parse_args()

seed = args.seed
mask_step = args.mask_step
bboxs = arg_to_bboxs(args.bboxs_arg)
num_frame = args.num_frame
height = args.height
width = args.width
fixRope_step = args.fixRope_step
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

with open(args.prompt_json, 'r') as f:
    prompts_data = json.load(f)
    prompt_item = prompts_data[0]
    fg_prompt = prompt_item["fg_prompt"]
    bg_prompt = prompt_item["bg_prompt"]
    base_prompt = prompt_item["base_prompt"]

print(f"Using seed: {seed}, mask_step: {mask_step}, fixRope_step: {fixRope_step}, bboxs: {bboxs}")
print(f"Using prompt: fg_prompt: {fg_prompt}, bg_prompt: {bg_prompt}, base_prompt: {base_prompt}")

output_path = args.output_path
output_path_withbox = args.output_path_withbox

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe = myWanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator=torch.Generator(device='cpu').manual_seed(seed)

attn_procs = {}
cnt = 0
for name in pipe.transformer.attn_processors.keys():
    cnt+=1
    attn_procs[name] = MyWanAttnProcessor2_0()
pipe.transformer.set_attn_processor(attn_procs)
print(f"******{cnt} attn_procs changed.*******")
pipe.scheduler = scheduler
device = 'cuda'
pipe.to(device)


latent_num_frame = num_frame//4 + 1
bbox_h = height // 16
bbox_w = width // 16

bbox_mask = torch.zeros([latent_num_frame, 1, bbox_h, bbox_w]).to(device)

# dynamic box
PATHS = plan_path(bboxs, video_length=num_frame)[::4]
assert latent_num_frame == len(PATHS), "latent_num_frame != len(PATHS)"
for i in range(latent_num_frame):
    h_start = int(PATHS[i][0] * bbox_h)
    h_end = int(PATHS[i][1] * bbox_h)
    w_start = int(PATHS[i][2] * bbox_w)
    w_end = int(PATHS[i][3] * bbox_w)
    bbox_mask[i, :, h_start:h_end, w_start:w_end] = 1


latents = None


encoder_attention_mask = torch.Tensor([False for i in range(512)] + [True for i in range(512)])
encoder_attention_mask = encoder_attention_mask.to(device)


output = pipe(
     prompt=base_prompt,
     negative_prompt=negative_prompt,
     height=height,
     width=width,
     num_frames=num_frame,
     guidance_scale=5,
     generator=generator,
     attention_kwargs={"bbox_mask": bbox_mask,"encoder_attention_mask":encoder_attention_mask,"bg_prompt":bg_prompt,"fg_prompt":fg_prompt,"fixRope_step":fixRope_step, "mask_step":mask_step},
     latents=latents,
    ).frames[0]

export_to_video(output, output_path, fps=args.fps)

if output_path_withbox is not None:
    save_videos_with_bbox(torch.Tensor(output).unsqueeze(0).unsqueeze(0).permute(0,1,5,2,3,4), output_path_withbox, fps=args.fps, input_traj=bboxs)