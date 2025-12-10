

def arg_to_bboxs(bboxs_str):
    bboxs_flat_str = bboxs_str.split(",")
    bboxs_flat = []
    for index, s in enumerate(bboxs_flat_str):
        if index % 5 == 0:
            bboxs_flat.append(int(s))
        else:
            bboxs_flat.append(float(s))
    bbox_len = 5 
    bboxs = [
        bboxs_flat[i:i+bbox_len] 
        for i in range(0, len(bboxs_flat), bbox_len)
    ]
    return bboxs

def bboxs_to_arg(bboxs):
    # 格式化bboxs为命令参数（将所有数值拼接为字符串，用逗号分隔）
    # 示例：[[0,0.5,0.8,0.2,0.5], [48,0.5,0.8,0.5,0.8]] → "0,0.5,0.8,0.2,0.5,48,0.5,0.8,0.5,0.8"
    bboxs_flat = [str(num) for bbox in bboxs for num in bbox]  # 展平列表并转为字符串
    bboxs_arg = ",".join(bboxs_flat)  # 拼接为单个字符串参数
    return bboxs_arg

def plan_path(input, video_length = 49):
    len_input = len(input)
    path = [input[0][1:]]
    for i in range(1, len_input):
        start = input[i-1]
        end = input[i]
        start_frame = start[0]
        end_frame = end[0]
        h_start_change = (end[1] - start[1]) / (end_frame - start_frame)
        h_end_change = (end[2] - start[2]) / (end_frame - start_frame)
        w_start_change = (end[3] - start[3]) / (end_frame - start_frame)
        w_end_change = (end[4] - start[4]) / (end_frame - start_frame)
        for j in range(start_frame+1, end_frame + 1):
            increase_frame = j - start_frame
            path += [[increase_frame * h_start_change + start[1], increase_frame * h_end_change + start[2], increase_frame * w_start_change + start[3], increase_frame * w_end_change + start[4]]]
 
    if input[0][0] > 0:
        h_change = path[1][0] - path[0][0]
        w_change = path[1][2] - path[0][2]
        for i in range(input[0][0]):
            path = [path[0][0] - h_change, path[0][1] - h_change, path[0][2] - w_change, path[0][3] - w_change] + path

    if input[-1][0] < video_length - 1:
        h_change = path[-1][0] - path[-2][0]
        w_change = path[-1][2] - path[-2][2]
        for i in range(video_length - 1 - input[-1][0]):
            path = path + [path[-1][0] + h_change, path[-1][1] + h_change, path[-1][2] + w_change, path[-1][3] + w_change]

    return path



import torch
import torch.fft as fft
import math
import torchvision
import os



def save_videos_with_bbox(batch_tensors, bbox_savepath, fps=10, input_traj=[]):
    # b,samples,c,t,h,w
    PATHS = plan_path(input_traj)
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        h_len = video.shape[3]
        w_len = video.shape[4]
        for i in range(video.shape[1]):
            single_video = video[:, i]
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in single_video] #[3, 1*h, n*w]
            grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
            # grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
            for j in range(video.shape[0]): 
                h_start = int(PATHS[j][0] * h_len)
                h_end = int(PATHS[j][1] * h_len)
                w_start = int(PATHS[j][2] * w_len)
                w_end = int(PATHS[j][3] * w_len)

                h_start = max(1, h_start)
                h_end = min(h_len-1, h_end)
                w_start = max(1, w_start)
                w_end = min(w_len-1, w_end)

                grid[j, h_start-1:h_end+1, w_start-1:w_start+2, :] = torch.ones_like(grid[j, h_start-1:h_end+1, w_start-1:w_start+2, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)
                grid[j, h_start-1:h_end+1, w_end-2:w_end+1, :] = torch.ones_like(grid[j, h_start-1:h_end+1, w_end-2:w_end+1, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)
                grid[j, h_start-1:h_start+2, w_start-1:w_end+1, :] = torch.ones_like(grid[j, h_start-1:h_start+2, w_start-1:w_end+1, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)
                grid[j, h_end-2:h_end+1, w_start-1:w_end+1, :] = torch.ones_like(grid[j, h_end-2:h_end+1, w_start-1:w_end+1, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)

            torchvision.io.write_video(bbox_savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})
