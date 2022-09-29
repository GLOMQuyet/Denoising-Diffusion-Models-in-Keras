# Denoising-Diffusion-Models-in-Keras
In this article we will go through the code from start to finish with DDPM using keras . In the test I will try with fashion_mnist.

Example 1:

![Alt Text](https://github.com/GLOMQuyet/Denoising-Diffusion-Models-in-Keras/blob/main/dm.gif)

Example 2:

![Alt Text](https://github.com/GLOMQuyet/Denoising-Diffusion-Models-in-Keras/blob/main/test2.gif)

## Instructions on how to use

```
Step 1:!git clone https://github.com/GLOMQuyet/Denoising-Diffusion-Models-in-Keras/tree/main
Step 2: Please create python env first
Step 3:!python display_image.py
```

## How to train model from scratch?

```
Step 1:!git clone https://github.com/GLOMQuyet/Denoising-Diffusion-Models-in-Keras/tree/main
Step 2: Please create python env first
Step 3:!python train.py
Step 4:!python display_image.py
```

Train custom data and operation more details please see on colab ![DM.ipynb](https://github.com/GLOMQuyet/Denoising-Diffusion-Models-in-Keras/blob/main/DM.ipynb)

## Citations

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

```bibtex
@article{Choi2022PerceptionPT,
    title   = {Perception Prioritized Training of Diffusion Models},
    author  = {Jooyoung Choi and Jungbeom Lee and Chaehun Shin and Sungwon Kim and Hyunwoo J. Kim and Sung-Hoon Yoon},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2204.00227}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Qiao2019WeightS,
    title   = {Weight Standardization},
    author  = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Loddon Yuille},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1903.10520}
}
```
