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

3. Run the code for basic testing, one GPU with 20GB memory is required for current code version. 

```shell
bash run.sh
```
Check the outpainted results from the `results' folder.



### Cite
```bibtex
@article{wang2024your,
  title={Be-Your-Outpainter: Mastering Video Outpainting through Input-Specific Adaptation},
  author={Wang, Fu-Yun and Wu, Xiaoshi and Huang, Zhaoyang and Shi, Xiaoyu and Shen, Dazhong and Song, Guanglu and Liu, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2403.13745},
  year={2024}
}
```
