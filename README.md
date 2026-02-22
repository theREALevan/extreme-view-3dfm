<div align="center">
<h1>Emergent Extreme-View Geometry <br> in 3D Foundation Models</h1>

<a href="https://arxiv.org/abs/2511.22686"><img src="https://img.shields.io/badge/arXiv-2511.22686-b31b1b" alt="arXiv"></a> &nbsp; <a href="https://cornell-vailab.github.io/Ext-3DFMs/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> &nbsp; <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face"></a>

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
| **π³** |  Coming soon |  Coming soon |
| **World-Mirror** |  Coming soon |  Coming soon |

> **Note:** Only VGGT evaluation is available in this release. The scripts below are for VGGT only.

---

## Evaluation

### Setup

1. **Clone the repository** (with submodules):

   ```bash
   git clone --recurse-submodules https://github.com/theREALevan/extreme-view-3dfm.git
   cd extreme-view-3dfm
   ```

2. **Install dependencies** following the [VGGT Quick Start](https://github.com/facebookresearch/vggt?tab=readme-ov-file#quick-start) guide.

### Run evaluation

1. In `scripts/eval.sh`, set `BASE_DIR` to your **MegaUnScene** data root.
2. Run:

   ```bash
   ./scripts/eval.sh
   ```

---

<div align="center">

*For more details, visit the [project page](https://cornell-vailab.github.io/Ext-3DFMs/).*

</div>
