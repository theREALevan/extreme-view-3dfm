<div align="center">
<h1>Emergent Extreme-View Geometry <br> in 3D Foundation Models</h1>

<a href="https://arxiv.org/abs/2511.22686"><img src="https://img.shields.io/badge/arXiv-2511.22686-b31b1b" alt="arXiv"></a> &nbsp;
<a href="https://cornell-vailab.github.io/Ext-3DFMs/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> &nbsp;
<a href="https://github.com/theREALevan/extreme-view-3dfm"><img src="https://img.shields.io/badge/Main_Code-black?logo=github" alt="Main Code Repository"></a> &nbsp;
<a href="https://github.com/jot-jt/extreme-view-3dfm-gen-eval"><img src="https://img.shields.io/badge/Generalization_Evaluations-black?logo=github" alt="Generalization Eval Repository"></a> &nbsp;
<a href="https://huggingface.co/datasets/cornell-vailab/megaunscene"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face"></a>

<p align="center"><em>CVPR 2026</em></p>

**Yiwen Zhang**¹ &nbsp; **Joseph Tung**² &nbsp; **Ruojin Cai**³ &nbsp; **David Fouhey**² &nbsp; **Hadar Averbuch-Elor**¹

¹Cornell University &nbsp; ²New York University &nbsp; ³Kempner Institute, Harvard University
</div>

```bibtex
@misc{zhang2025emergentextremeviewgeometry3d,
      title={Emergent Extreme-View Geometry in 3D Foundation Models}, 
      author={Yiwen Zhang and Joseph Tung and Ruojin Cai and David Fouhey and Hadar Averbuch-Elor},
      year={2025},
      eprint={2511.22686},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.22686}, 
}
```

---

## Code Release

| Component | Evaluation | Training |
|:----------|:----------:|:--------:|
| **VGGT** | ✅ Released |  Coming soon |
| **π³** | ✅ Released |  Coming soon |
| **World-Mirror (WM)** | ✅ Released |  Coming soon |

---

## Extreme Relative Rotation Estimation Evaluation

### Setup

1. **Clone the repository** (with submodules):

   ```bash
   git clone --recurse-submodules https://github.com/theREALevan/extreme-view-3dfm.git
   cd extreme-view-3dfm
   ```

2. **Install dependencies:** Go into the corresponding model folder under `models/` and follow that repo’s setup.  
   - **VGGT:** `models/vggt` — see [VGGT Quick Start](https://github.com/facebookresearch/vggt?tab=readme-ov-file#quick-start).  
   - **π³ (Pi3):** `models/pi3` — see [Pi3 Quick Start](https://github.com/yyfz/Pi3?tab=readme-ov-file#-quick-start).  
   - **World-Mirror (WM):** `models/worldmirror` — see [Dependencies and Installation](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror#%EF%B8%8F-dependencies-and-installation).

### Run evaluation

1. In `scripts/eval.sh`, set `BASE_DIR` to your **MegaUnScene** data root.
2. Set `MODEL` to `vggt`, `pi3`, or `wm`. Set `NO_CKPT=1` to use the pre-trained model only; otherwise the fine-tuned checkpoint is used.
3. Run:

   ```bash
   ./scripts/eval.sh
   ```

## Generalization Evaluations
For evaluations on monocular depth, multiview pose estimation, and dense reconstruction (including on UnSceneRecon), please refer to the [generalization evaluation GitHub repo](https://github.com/jot-jt/extreme-view-3dfm-gen-eval).

## MegaUnScene Dataset
For MegaUnScene information and download instructions, please refer to the [HuggingFace dataset page](https://huggingface.co/datasets/cornell-vailab/megaunscene).

---

<div align="center">

*For more details, visit the [project page](https://cornell-vailab.github.io/Ext-3DFMs/).*

</div>
