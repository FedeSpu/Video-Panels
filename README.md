# [CVPR 2026] Video Panels for Long Video Understanding

*Official implementation of the paper "Video Panels for Long Video 
Understanding".*

[**Lars Doorenbos**](https://scholar.google.com/citations?user=i2LqZCwAAAAJ&hl=en)\*<sup>1,2</sup> &nbsp;.&nbsp; [**Federico Spurio**](https://github.com/FedeSpu)\*<sup>1,2</sup> &nbsp;.&nbsp; [**Juergen Gall**](https://scholar.google.de/citations?user=1CLaPMEAAAAJ)<sup>1,2</sup>
<br>
<small><sup>1</sup>University of Bonn, <sup>2</sup>Lamarr Institute for Machine Learning and Artificial Intelligence</small>
<br>
<small>\*Equal Contribution</small>

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2509.23724)
[![Project Page Badge](https://img.shields.io/badge/Project-VideoPanels-Green)](https://fedespu.github.io/Video-Panels/)

---


## Overview
**Video Panels** is a training-free, parameter-free, and model-agnostic approach designed to increase the temporal coverage of existing Video Language Models (VLMs). 

The underlying idea is straightforward, as illustrated below: we trade-off spatial resolution for temporal resolution by combining multiple sequential frames into single composite images, or *panels*. 

<p align="center">
  <img src="./images/Teaser.png" alt="Video Panels Teaser" width="80%" height="auto"/>
</p>

## Requirements

The provided implementation works with `decord` as the video decoding backend. We have tested our code with the following environment:

*   **Python:** 3.10.8
*   **decord:** 0.6.0
*   **opencv-python:** 4.11.0.86

You can install the required packages via pip:

```bash
pip install decord==0.6.0 opencv-python==4.11.0.86
```

## Repository Structure

Click on the filenames below to jump directly to the source code:

*   **[`paneling.py`](./paneling.py)**: Contains the core functions and logic for paneling input videos.
*   **[`class_paneling.py`](./class_paneling.py)**: Provides an example demonstrating how to integrate and use the paneling functions within a class structure.

Inside the **[`lmms_eval/models/`](./lmms_eval/models/)** directory, we provide examples for integrating with [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) (branch `8895505b3fb087bdfc91cb5f0a1b3a6a6a0c0914`):

*   **[`llava_onevision.py`](./lmms_eval/models/llava_onevision.py)**: Example integration for evaluating the [LLaVA OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) (branch `bcba57a8441d74dd81636a8364953b47b5f1e9be`) model. 

## Usage

### 1. Control Parameters

You can personalize the panel generation by adjusting the following parameters:

* **`panel_width` (α):** Number of frames to combine along the horizontal axis (`panel_width=2` in the paper).
* **`panel_height` (β):** Number of frames to combine along the vertical axis (`panel_height=2` in the paper).
* **`fps_limit` (proportional to γ):** Controls the temporal stride (γ) between sampled frames. It dictates how many seconds of footage must pass between one sampled frame and the next. 
  * *Example:* In a 30 FPS video, `fps_limit=1` ensures at least 30 frames between samples. `fps_limit=2` ensures at least 60 frames, and so on.
* **`border_px`:** Number of black pixels to add as a border around every paneled frame. Set `border_px=0` to reproduce the exact numbers from the paper.
    
* **`verbose`:** Toggles console output, informing the user whether the video was paneled successfully.
* **`plot_video`:** Enables plotting of the output video as an image grid for visual debugging.

### 2. Usage in `lmms-eval`

Integrating Video Panels into the existing evaluation pipeline is simple:

1. Replace the corresponding model inside the original `lmms-eval/lmms_eval/models` directory with the files provided in this repository ([`lmms_eval/models`](./lmms_eval/models/)). 
2. Add the paneling parameters to the `--model_args` flag when launching `lmms_eval`.

**Example: Evaluating LLaVA OneVision on VideoMME:**

```bash
accelerate launch --num_processes=4 --main_process_port 12399 -m lmms_eval \
    --model llava_onevision \
    --model_args max_frames_num=32,panel_width=2,panel_height=2,fps_limit=1\
    --tasks videomme \
    --batch_size 1 \
```

---

## Citation

If you find this work helpful, consider citing it using

```
@article{doorenbos2025video,
  title={Video Panels for Long Video Understanding},
  author={Doorenbos, Lars and Spurio, Federico and Gall, Juergen},
  journal={Computer Vision and Pattern Recognition},
  year={2026}
}
```