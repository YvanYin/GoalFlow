<p align="center">
    <img alt="goalflow_logo" src="assets/goalflow_logo.png" width="500">
    <!-- <h1 align="center">A Framework for Vehicle Motion Planning Research</h1> -->
    <h3 align="center"><a href="https://arxiv.org/abs/2503.05689">Paper</a> | <a href="#weight">Weight</a> | <a href="#cache">Cache</a> | <a href="#poster">Poster</a> </h3>
</p>

<br/>

> [**GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation
in End-to-End Autonomous Driving**](https://arxiv.org/abs/2503.05689)  <br>
> [Zebin Xing](https://github.com/ZebinX)<sup>1,2*</sup>, [Xingyu Zhang]()<sup>2*</sup>, [Yang Hu]()<sup>1,2</sup>, [Bo Jiang]()<sup>4,2</sup>, [Tong He](https://tonghe90.github.io/)<sup>5</sup>, [Qian Zhang]()<sup>2</sup>, [Xiaoxiao Long](https://www.xxlong.site/)<sup>3</sup>, [Wei Yin](https://yvanyin.net/)<sup>2✝</sup>  <br>
> <sup>1</sup> University of Chinese Academy of Sciences, <sup>2</sup> Horizon Robotics, <sup>3</sup> Nanjing University, <sup>4</sup> Huazhong University of Science & Technology, <sup>3</sup> Shanghai AI Laboratory  <br>
> <br>
> Computer Vision and Pattern Recognition (CVPR), 2025 <br>
>
This the official repo of GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving (CVPR 2025). We provide a publicly accessible configuration for validation, model weights and an optimized version for the NavSim testing. (Code is building...)

<br/>

## News
* **`20 Mar, 2025`:**  We released our paper on [arXiv](https://arxiv.org/abs/2503.05689). Code is coming soon.
* **`27 Feb, 2025`:**  GoalFlow was accepted at [CVPR](https://cvpr.thecvf.com/Conferences/2025)!


<br/>

## Introduction
> In autonomous driving, multiple optimal trajectories exist, like overtaking or following. (1) Traditional methods efficiently generate safe trajectories but struggle with multimodal ones. (2) Generative methods like diffusion models capture multimodal distributions but require heavy hardware and prior information. We propose GoalFlow, a goal-point-based method that guides trajectory planning. With a map-free evaluation and an efficient diffusion variant, Flow Matching, we reduce inference steps, achieving superior performance with just one denoising step.

<div align="center">
<img src="./assets/main_fig.png" />
</div>

## Visualization
### Goal Point Distribution
From left to right, they are respectively the distributions of DAC, distance, and the final score.
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
    <img src="./assets/visual_goal_point/dac_scores/0a44947ca9e85579.png" width="200" />
    <img src="./assets/visual_goal_point/im_scores/0a44947ca9e85579.png" width="200" />
    <img src="./assets/visual_goal_point/final_scores/0a44947ca9e85579.png" width="200" />
    <img src="./assets/visual_goal_point/dac_scores/2a06b778a64b545e.png" width="200" />
    <img src="./assets/visual_goal_point/im_scores/2a06b778a64b545e.png" width="200" />
    <img src="./assets/visual_goal_point/final_scores/2a06b778a64b545e.png" width="200" />
    <img src="./assets/visual_goal_point/dac_scores/7abf60c1594953cf.png" width="200" />
    <img src="./assets/visual_goal_point/im_scores/7abf60c1594953cf.png" width="200" />
    <img src="./assets/visual_goal_point/final_scores/7abf60c1594953cf.png" width="200" />
</div>

### Comparison with Other Methods

<div align="center">
    <img src="./assets/visualization.png" />
</div>

### Gifs for Driving
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
    <img src="./assets/25b719c231d85e56.gif" width="200" />
    <img src="./assets/1a1fbb255ec55813.gif" width="200" />
    <img src="./assets/41da8ea7c14754d2.gif" width="200" />
    <img src="./assets/d2440edd19d954b5.gif" width="200" />
    <img src="./assets/fb0a26a28ec359ce.gif" width="200" />
    <img src="./assets/cf12097663665430.gif" width="200" />
</div>


## Results
Planning results on the proposed **NAVSIM** **Test** benchmark. Please refer to the [paper](https://arxiv.org/abs/2503.05689) for more details.

| Method           | S<sub>NC</sub> ↑ | S<sub>DAC</sub> ↑ | S<sub>TTC</sub> ↑ | S<sub>CF</sub> ↑ | S<sub>EP</sub> ↑ | S<sub>PDM</sub> ↑ |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Constant Velocity | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP   | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| LTF             | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| TransFuser      | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| UniAD          | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive     | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| **GoalFlow (Ours)** | **98.4** | **98.3** | **94.6** | **100** | **85.0** | **90.3** |
| *GoalFlow<sup>†</sup>* | *99.8* | *97.9* | *98.6* | *100* | *85.4* | *92.1* |
| *Human<sup>‡</sup>* | *100* | *100* | *100* | *99.9* | *87.5* | *94.8* |

## To Do
- \[ \] Code for goal point construction module 
- \[ \] Goal Point scorer and cluster vocabulary cache
- \[ \] Weights of model
- \[ \] Code for validation
- \[ \] Tutorial for installation
- \[√\] Initial repo & main paper


## Getting started
The code is building...

## Contact
If you have any questions or suggestions, please feel free to open an issue or contact us (xzebin@bupt.edu.cn).

## Acknowledgement
<p>1. We have gained valuable insights from <a href="https://arxiv.org/abs/2406.06978" target="_blank">Hydra-MDP</a>, which provided many inspiring ideas referenced in our work.</p>
<p>2. We referred to an excellent GitHub project, <a href="https://github.com/autonomousvision/tuplan_garage" target="_blank">tuplan garage</a>, and incorporated aspects of its page design.</p>

<p>3. GoalFlow is also greatly inspired by the following outstanding contributions to the open-source community:</p>
<ul>
    <a href="https://github.com/autonomousvision/navsim" target="_blank">NAVSIM</a> | <a href="https://github.com/autonomousvision/transfuser" target="_blank">TransFuser</a> | <a href="https://github.com/hustvl/VAD" target="_blank">Diffusion-ES</a> | <a href="" target="_blank">VAD-v2</a>
</ul>



## Citation
If you find GoalFlow useful, please consider giving us a star &#127775; and citing our paper with the following BibTeX entry.

```BibTeX
@misc{xing2025goalflowgoaldrivenflowmatching,
      title={GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving}, 
      author={Zebin Xing and Xingyu Zhang and Yang Hu and Bo Jiang and Tong He and Qian Zhang and Xiaoxiao Long and Wei Yin},
      year={2025},
      eprint={2503.05689},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.05689}, 
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
