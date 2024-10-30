# Be-Your-Outpainter

![diversity](https://github.com/G-U-N/Be-Your-Outpainter/assets/60997859/0bd3e853-0aa3-4fbc-8e19-32258f39c4c2)


https://github.com/G-U-N/Be-Your-Outpainter/assets/60997859/9c10fde7-c881-4dd6-89b3-00b95ef9d099


### Run

1. Install Environment

```shell
conda env create -f environment.yml
```
2. Downloads

Download the models folder from Huggingface.

```shell
git clone https://huggingface.co/wangfuyun/Be-Your-Outpainter
```

3. Run the code for basic testing. Single GPU with 20GB memory is required for current code version. Reduce the video length if GPU memory is limited. 

```shell
bash run.sh
```
Check the outpainted results from the `results' folder.

### Outpaint Your Own Videos
Edit the [exp.yaml](https://github.com/G-U-N/Be-Your-Outpainter/blob/master/configs/exp.yaml) to outpaint your own videos.
```yaml
exp: # Name of your task

  train_data:
    video_path: "data/outpaint_videos/SB_Dog1.mp4"                            # source video path
    prompt: "a cute dog, garden, flowers"                                     # source video prompts for tuning
    n_sample_frames: 16                                                       # source video length
    width: 256                                                                # source video width
    height: 256                                                               # source video height
    sample_start_idx: 0                                                       # set to 0 by default. Sampling frames from the beginning of the video
    sample_frame_rate: 1                                                      # fps of video 
  
  validation_data:
    prompts:
      - "a cute dog, garden, flowers"                                         # prompts applied for outpainting. 
    prompts_l:
      - "wall"
    prompts_r:
      - "wall"
    prompts_t:
      - ""
    prompts_b:
      - ""

    prompts_neg:
      - ""


    is_grid: False                                                            # set as True to enable prompts_r, prompts_l, prompts_t, prompts_b 
    video_length: 16                                                          # video length. The same as in the train_data config
    width: 256
    height: 256

    scale_l: 0
    scale_r: 0
    scale_t: 0.5                                                              # How to expand the video field. For a 512x512 source video. Set scale_l and scale_r to 0.5, and it will generate 512x(512 + 512 * 0.5 + 512 * 0.5) = 512 x 1024 video.
    scale_b: 0.5

    window_size: 16                                                           # only used in longer video outpainting
    stride: 4


    repeat_time: 0                                                            # set to 4 enable noise regret
    jump_length: 3

    num_inference_steps: 50                                                   # inference steps for outpainting
    guidance_scale: 7.5             


    bwd_mask: null                                                            # not applied
    fwd_mask: null
    bwd_flow: null
    fwd_flow: null

    warp_step: [0,0.5]
    warp_time: 3

  mask_config:                                                                # how to set mask for tuning
    mask_l: [0., 0.4]
    mask_r: [0., 0.4]
    mask_t: [0., 0.4]
    mask_b: [0., 0.4]
```

### Evaluation
Please check [metric.py](https://github.com/G-U-N/Be-Your-Outpainter/blob/master/src/utils/metrics.py).

Important notes:
- The metrics are reported as the average value of the settings with mask ratio 0.25 and mask ratio 0.66. For example, if the PSNR of mask ratio 0.25 is 30 and the PSNR of the mask ratio 0.66 is 20, and then the reported PSNR is (20+30)/2 = 25.
- The VAE might suffer from decoding losses. We replace the known region with the GT for evaluation.

### Cite
```bibtex
@article{wang2024your,
  title={Be-Your-Outpainter: Mastering Video Outpainting through Input-Specific Adaptation},
  author={Wang, Fu-Yun and Wu, Xiaoshi and Huang, Zhaoyang and Shi, Xiaoyu and Shen, Dazhong and Song, Guanglu and Liu, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2403.13745},
  year={2024}
}
```
