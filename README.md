<div align="center">
<h2>DiTraj: Training-free Trajectory Control For Video Diffusion Transformer</h2>

[Cheng Lei](https://github.com/leichengjiayou)<sup>12†</sup>, [Jiayu Zhang](https://github.com/xduzhangjiayu)<sup>2†‡</sup>, [Yue Ma](https://mayuelala.github.io/)<sup>3*</sup>, [Xinyu Wang]()<sup>4</sup>, [Long Chen]()<sup>2</sup>, [Liang Tang]()<sup>2</sup>, [Yiqiang Yan]()<sup>2</sup>, [Fei Su]()<sup>1</sup>, [Zhicheng Zhao]()<sup>1*</sup>

<sup>1</sup> Beijing University of Posts and Telecommunications,  <sup>2</sup> Lenovo,  <sup>3</sup> HKUST,  <sup>4</sup> Tsinghua University

†Equal Contribution &emsp; ‡ Project Lead &emsp; *Corresponding Author


<a href='https://xduzhangjiayu.github.io/DiTraj_Project_Page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
[![arXiv](https://img.shields.io/badge/arXiv-2509.21839-b31b1b.svg)](https://arxiv.org/abs/2509.21839)
[![GitHub Stars](https://img.shields.io/github/stars/xduzhangjiayu/DiTraj)](https://github.com/xduzhangjiayu/DiTraj)


<div align="left">
We propose DiTraj, the first training-free trajectory control framework for DiT-based video generation model. Given
an input bbox trajectory guidance, DiTraj enables generating high-quality videos that align with the target trajectory. Our method achieves state-of-the-art performance in both video quality and trajectory controllability. It can be adapted to most DiT-based video generation models (Wan2.1, CogVideoX etc.).

<div align="left">

# 🎇 Showcase
<table class="center">
  <td><img src=teaser/2.gif width="256"></td>
  <td><img src=teaser/5.gif width="256"></td>
  <td><img src=teaser/6.gif width="256"></td>
</table >

<table class="center">
  <td><img src=teaser/7.gif width="256"></td>
  <td><img src=teaser/8.gif width="256"></td>
  <td><img src=teaser/9.gif width="256"></td>
</table >

<table class="center">
  <td><img src=teaser/1.gif width="256"></td>
  <td><img src=teaser/4.gif width="256"></td>
  <td><img src=teaser/3.gif width="256"></td>
</table >

# 🎇 Complex Trajectory
<table class="center">
  <td><img src=teaser/com1.gif width="256"></td>
  <td><img src=teaser/com2.gif width="256"></td>
  <td><img src=teaser/com3.gif width="256"></td>
</table >

<table class="center">
  <td><img src=teaser/com4.gif width="256"></td>
  <td><img src=teaser/com5.gif width="256"></td>
  <td><img src=teaser/com6.gif width="256"></td>
</table >


For more examples, please refer to our project page (https://xduzhangjiayu.github.io/DiTraj_Project_Page/).

<div align="left">



# 📖 Pipeline
<p>
<div align="center">
<img src="teaser/method.png" width="1080px"/>
<div>
<div align="left">
<div>
<div>
<div align="left">
<div>
<div>

# 🔥 News
[2025.9.29] Paper released!  
[2025.12.10] Code released!

# 👨‍💻 ToDo
- [x] Release Paper on arxiv
- [x] Release Code
- [ ] Release Gradio demo with user-friendly interaction

# 🚀 Getting Started
## Environment Requirement
Clone the repo:
```
git clone https://github.com/xduzhangjiayu/DiTraj.git
```
Then:
```
conda create --name DiTraj python=3.11
conda activate DiTraj
pip install -r requirements.txt
git clone --branch v0.33.1 https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```
Finally:  
Replace the `./module/transformer_wan.py` file in the `./diffusers/src/diffusers/models/transformers/transformer_wan.py`
## Generate your own video!  
1. First, input your prompts in the `test_prompts.txt.`   
2. Then, run the following command:    
```
python prompt_extend.py (optional)
```
```
python prompt_refine.py
```
 `demo/test_prompts_refined.json` will be generated, including the bg/fg prompt.   
 
3. Define your trajectory in run.py (line 16)
You can set the bbox in several keyframes , (x1,y1) is the bbox top left corner, (x2,y2) is the bottom right corner. Each keyframe uses [frame_id, y1, y2, x1, x2]
For example:
```
bboxs = [
            [0, 0.3, 0.7, 0.1, 0.4], # frame 0: Left side
            [80, 0.3, 0.7, 0.7,1.0]  # frame 80: Right side
        ]
```
if you want to use a complex trajectory, you can use the following code:
```
bboxs = [
            [0, 0.05, 0.55, 0.05, 0.45], # frame 0: Top-left
            [20, 0.05, 0.55, 0.55, 0.95], # frame 20: Top-right
            [40, 0.45, 0.95, 0.55, 0.95], # frame 40: Bottom-left
            [60, 0.45, 0.95, 0.05, 0.45], # frame 80: Bottom-right
            [80, 0.05, 0.55, 0.05, 0.45], # frame 80: Top-left 
        ]
```
4. Run the following command:  
```
python run.py
```
5. The video will be saved in the in `demo/output.mp4` and `demo/output_box.mp4` (video with bbox)  

<div>
<div>

# 📚 Acknowledgements
Our codebase builds on [diffusers](https://github.com/huggingface/diffusers), thanks for the great work!

# 🖋️ Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!
```
@misc{lei2025ditrajtrainingfreetrajectorycontrol,
      title={DiTraj: training-free trajectory control for video diffusion transformer}, 
      author={Cheng Lei and Jiayu Zhang and Yue Ma and Xinyu Wang and Long Chen and Liang Tang and Yiqiang Yan and Fei Su and Zhicheng Zhao},
      year={2025},
      eprint={2509.21839},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.21839}, 
}
```

# License
This code is licensed under CC BY-NC 4.0 and intended for research use only — no commercial use allowed.