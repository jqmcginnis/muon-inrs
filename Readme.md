## Code for Optimizing Rank for High-Fidelity Implicit Neural Representations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1g918SrErj66Sktx5j6ypQr2dNKB5gS7b/view?usp=sharing)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://muon-inrs.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.14366-b31b1b.svg)](https://arxiv.org/abs/2512.14366)

Implicit Neural Representations (INRs) based on vanilla Multi-Layer Perceptrons (MLPs) are widely believed to be incapable of representing high-frequency content. This has directed research efforts towards architectural interventions, such as coordinate embeddings or specialized activation functions, to represent high-frequency signals. In this paper, we challenge the notion that the low-frequency bias of vanilla MLPs is an intrinsic, architectural limitation to learn high-frequency content, but instead a symptom of stable rank degradation during training. We empirically demonstrate that regulating the network's rank during training substantially improves the fidelity of the learned signal, rendering even simple MLP architectures expressive. Extensive experiments show that using optimizers like Muon, with high-rank, near-orthogonal updates, consistently enhances INR architectures even beyond simple ReLU MLPs. These substantial improvements hold across a diverse range of domains, including natural and medical images, and novel view synthesis, with up to 9 dB PSNR improvements over the previous state-of-the-art.

<img src="assets/opening_figure.drawio.png" width="50%" alt="Opening Figure">

## Using Muon for training INRs

Muon (Jordan et al.) is typically used for the **hidden layers only**. Thus, instead of using a single parameter group, we define two parameter groups and pass them to
the SingleDeviceMuonWithAuxAdam. We recommend tryining a range of learning rates for both Muon (hidden weights) and Adam (auxilary, for bias and input/output weights.)

```
    # conventional optimization
    optim_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)

    # optimizing with muon
    muon_params = []
    other_params = []
    for name, p in model_muon.named_parameters():
        if p.ndim == 2 and p.size(0) > 1 and p.size(1) > 1:
            muon_params.append(p)
        else:
            other_params.append(p)

    optim_muon = SingleDeviceMuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=0.0),
        dict(params=other_params, use_muon=False, lr=aux_adam_lr, betas=(0.9, 0.999), weight_decay=0.0)
    ])

    # we recommend using a learning rate scheduler, similar to how INRs are typically trained with Adam
    scheduler_muon = torch.optim.lr_scheduler.CosineAnnealingLR(optim_muon, T_max=steps)
```

## Giving Muon a try

To easily checkout Muon, we provide a reference notebook here. Make a copy to get started.

### To replicate the experiments of our paper, we provide the following instructions.

#### 1. Setup and Data

#### Obtaining the Data
Please use the following links to obtain the necessary datasets:

| Experiment | Dataset Source |
| :--- | :--- |
| **Image Reconstruction** | [Kodak Image Dataset](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset) |
| **Audio Reconstruction** | [Audio Data from Sitzmann et al.](https://github.com/vsitzmann/siren/tree/master/data) |
| **CT Reconstruction** | [CT Data from Saragadam et al.](https://www.dropbox.com/scl/fo/s1q0a8uwvz0guii1lvglv/ADYHh-Og_p52DN08MJ3QUg8?rlkey=sceq8f7bys28yimdmeaki2gsu&e=1&dl=0) |
| **SISR Image Data** | [DIV2K Dataset Kaggle](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images) |
| **3D Shapes** | [Stanford Dataset](https://graphics.stanford.edu/data/3Dscanrep/) |
| **NeRF Dataset** | [Synthetic NeRF Dataset](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4) |

#### Environment Setup

**1. Standard Dependencies**
You require a standard PyTorch environment (tested on PyTorch 1.13+), along with script-specific libraries for data loading (e.g., `librosa` for audio, `pydicom`/`mrcfile` for CT). These can be installed via uv/pip/conda. Please note that NeRF and Shape Reconstruction are based on other repositories, and include installation instructions. 

**2. Installing Muon**
We use the Muon optimizer (Jordan et al.). Please install it via:
```bash
pip install git+https://github.com/KellerJordan/Muon
```
Note on PyTorch Versions: Muon utilizes the .mT syntax established in PyTorch 1.10. If you must use an older PyTorch version (e.g. for torch-ngp), please substitute all .mT with .transpose(-2, -1) in the function `zeropower_via_newtonschulz5` in muon.py.

#### 2. Overfitting Experiments
For specific learning rates and hyperparameters, please refer to the tables provided in the Supplementary Material.

Image Reconstruction: To employ Muon, specify `--optimizer muon` and set the learning rates for Muon and Adam parameter groups respectively.

```Bash
python3 fit_image.py \
  --image data/kodim01.png \
  --optimizer muon \
  --muon_lr <MUON_LR> \
  --lr <ADAM_LR>
```
Audio Reconstruction: Specify the audio file (e.g., gt_bach.wav) via `--audio`.

```Bash
python3 fit_audio.py \
  --audio data/gt_bach.wav \
  --optimizer muon \
  --muon_lr <MUON_LR> \
  --lr <ADAM_LR>
 ```

3D Shapes (SDF): We base our experiments on the FINER/BACON codebase. We will release a cleaned-up fork soon.

#### 3. Inverse Problems
CT Reconstruction: Run the reconstruction script specifying the image file path via --image.

```Bash
python3 fit_ct.py \
  --image data/ct_scan.dcm \
  --optimizer muon \
  --muon_lr <MUON_LR> \
  --lr <ADAM_LR>
```
Single Image Super-Resolution (SISR): Run the SISR script specifying the high-res image path via --image.

```Bash
python3 fit_sisr.py \
  --image data/div2k/0001.png \
  --optimizer muon \
  --muon_lr <MUON_LR> \
  --lr <ADAM_LR>
```
NeRF: We base our experiments on the FINER/Torch-NGP codebase. We will release a cleaned-up fork soon.

### Citation

If you find this work useful, please cite it as follows:

```bibtex
@misc{mcginnis2025optimizingrankhighfidelityimplicit,
      title={Optimizing Rank for High-Fidelity Implicit Neural Representations}, 
      author={Julian McGinnis and Florian A. Hölzl and Suprosanna Shit and Florentin Bieder and Paul Friedrich and Mark Mühlau and Björn Menze and Daniel Rueckert and Benedikt Wiestler},
      year={2025},
      eprint={2512.14366},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2512.14366](https://arxiv.org/abs/2512.14366)}, 
}
```

When using Muon in your work, please cite:

```bibtex
@misc{jordan2024muon,
      title={Muon: An optimizer for hidden layers in neural networks},
      author={Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein},
      year={2024},
      url={[https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)}
}
```
