# DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models
[Chong Mou](https://scholar.google.com/citations?user=SYQoDk0AAAAJ&hl=zh-CN),
[Xintao Wang](https://xinntao.github.io/),
[Jiechong Song](),
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ),
[Jian Zhang](https://jianzhang.tech/)

[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://mc-e.github.io/project/DragonDiffusion/)
[![arXiv](https://img.shields.io/badge/ArXiv-2304.08465-brightgreen)](https://arxiv.org/abs/2307.02421)

---

<p align="center">
  <img src="assets/teaser.png" height=300>
</p>

<div align="center">
DragonDiffusion</span> enables various editing modes for the real images, including object moving, object resizing, object appearance replacement, and content dragging.
</div>

## Updates

- [2023/7/6] Paper is available [here](https://arxiv.org/abs/2307.02421).

---

## Introduction
In this paper, we aim to develop a fine-grained image editing scheme based on the strong correspondence of intermediate features in diffusion models. To this end, we design a classifier-guidance-based method to transform the editing signals into gradients via feature correspondence loss to modify the intermediate representation of the diffusion model. The feature correspondence loss is designed with multiple scales to consider both semantic and geometric alignment. Moreover, a cross-branch self-attention is added to maintain the consistency between the original image and the editing result.

## Main Features

- Object Moving & Resizing
<p align="center">
  <img src="assets/res_move.png" height=250>
</p>

- Object Appearance Replacement
<p align="center">
  <img src="assets/res_app.png" height=250>
</p>

- Content Dragging
<p align="center">
  <img src="assets/res_drag.png" height=250>
</p>

## Related Works
[1] <a href="https://github.com/XingangPan/DragGAN">Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold</a>
</p>
<p>
[2] <a href="https://yujun-shi.github.io/projects/dragdiffusion.html">DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing</a> (The first attempt and presentation for point dragging on diffusion)
</p>
<p>
[3] <a href="https://dave.ml/selfguidance/">Diffusion Self-Guidance for Controllable Image Generation</a>
</p>
