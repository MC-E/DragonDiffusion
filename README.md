# [DragonDiffusion](https://arxiv.org/abs/2307.02421) + [DiffEditor](https://arxiv.org/abs/2402.02583)
[Chong Mou](https://scholar.google.com/citations?user=SYQoDk0AAAAJ&hl=zh-CN),
[Xintao Wang](https://xinntao.github.io/),
[Jiechong Song](),
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ),
[Jian Zhang](https://jianzhang.tech/)

[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://mc-e.github.io/project/DragonDiffusion/)
[![arXiv](https://img.shields.io/badge/ArXiv-2304.08465-brightgreen)](https://arxiv.org/abs/2307.02421)
[![arXiv](https://img.shields.io/badge/ArXiv-2402.02583-brightgreen)](https://arxiv.org/abs/2402.02583)

---
https://user-images.githubusercontent.com/54032224/302051504-dac634f3-85ef-4ff1-80a2-bd2805e067ea.mp4

## 🚩 **New Features/Updates**
- [2024/02/26] **DiffEditor** is accepted by CVPR 2024.
- [2024/02/05] Releasing the paper of **DiffEditor**.
- [2024/02/04] Releasing the code of **DragonDiffusion** and **DiffEditor**.
- [2024/01/15] **DragonDiffusion** is accepted by ICLR 2024 (**Spotlight**).
- [2023/07/06] Paper of **DragonDiffusion** is available [here](https://arxiv.org/abs/2307.02421).

---

# Introduction
**DragonDiffusion** is a turning-free method for fine-grained image editing. The core idea of DragonDiffusion comes from [score-based diffusion](https://arxiv.org/abs/2011.13456). It can perform various editing tasks, including object moving, object resizing, object appearance replacement, content dragging, and object pasting. **DiffEditor** further improves the editing accuracy and flexibility of DragonDiffusion.

# 🔥🔥🔥 Main Features  
### **Appearance Modulation**  
Appearance Modulation can change the appearance of an object in an image. The final appearance can be specified by a reference image.

<p align="center">
  <img src="https://huggingface.co/Adapter/DragonDiffusion/resolve/main/asserts/appearance.PNG" height=240>
</p>

### **Object Moving & Resizing**  
Object Moving can move an object in the image to a specified location.

<p align="center">
  <img src="https://huggingface.co/Adapter/DragonDiffusion/resolve/main/asserts/move.PNG" height=220>
</p>

### **Face Modulation**  
Face Modulation can transform the outline of one face into the outline of another reference face.

<p align="center">
  <img src="https://huggingface.co/Adapter/DragonDiffusion/resolve/main/asserts/face.PNG" height=250>
</p>

### **Content Dragging**  
Content Dragging can perform image editing through point-to-point dragging.

<p align="center">
  <img src="https://huggingface.co/Adapter/DragonDiffusion/resolve/main/asserts/drag.PNG" height=230>
</p>

### **Object Pasting**  
Object Pasting can paste a given object onto a background image.

<p align="center">
  <img src="https://huggingface.co/Adapter/DragonDiffusion/resolve/main/asserts/paste.PNG" height=250>
</p>

# 🔧 Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.1](https://pytorch.org/)
```bash
pip install -r requirements.txt
pip install dlib==19.14.0
```

# ⏬ Download Models 
All models will be automatically downloaded. You can also choose to download manually from this [url](https://huggingface.co/Adapter/DragonDiffusion).

# 💻 How to Test
Inference requires at least `16GB` of GPU memory for editing a `768x768` image.  
We provide a quick start on gradio demo.
```bash
python app.py
```

# Related Works
[1] <a href="https://github.com/XingangPan/DragGAN">Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold</a>
</p>
<p>
[2] <a href="https://yujun-shi.github.io/projects/dragdiffusion.html">DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing</a>
</p>
<p>
[3] <a href="https://arxiv.org/abs/2306.03881">
Emergent Correspondence from Image Diffusion</a></p>
<p>
[4] <a href="https://dave.ml/selfguidance/">Diffusion Self-Guidance for Controllable Image Generation</a>
</p>
<p>
[5] <a href="https://browse.arxiv.org/abs/2308.06721">IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models</a>
</p>

# 🤗 Acknowledgements
We appreciate the foundational work done by [score-based diffusion](https://arxiv.org/abs/2011.13456) and [DragGAN](https://arxiv.org/abs/2305.10973).

# BibTeX

    @article{mou2023dragondiffusion,
      title={Dragondiffusion: Enabling drag-style manipulation on diffusion models},
      author={Mou, Chong and Wang, Xintao and Song, Jiechong and Shan, Ying and Zhang, Jian},
      journal={arXiv preprint arXiv:2307.02421},
      year={2023}
    }
    @article{mou2023diffeditor,
      title={DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing},
      author={Mou, Chong and Wang, Xintao and Song, Jiechong and Shan, Ying and Zhang, Jian},
      journal={arXiv preprint arXiv:2402.02583},
      year={2023}
    }
