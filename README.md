# EMusicGen
[![license](https://img.shields.io/badge/License-LGPL-yellow.svg)](https://github.com/monetjoe/EMusicGen/blob/master/LICENSE)
[![Python application](https://github.com/monetjoe/EMusicGen/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/EMusicGen/actions/workflows/python-app.yml)
[![ds](https://img.shields.io/badge/ModelScope-Dataset-624aff.svg)](https://www.modelscope.cn/datasets/monetjoe/EMusicGen)
[![model](https://img.shields.io/badge/ModelScope-Model-624aff.svg)](https://www.modelscope.cn/models/monetjoe/EMusicGen)
[![demo](https://img.shields.io/badge/ModelScope-Demo-624aff.svg)](https://www.modelscope.cn/studios/monetjoe/EMusicGen)

Emotionally Conditioned Melody Generation in ABC Notation

![](./figs/model.jpg)

## Environment
```bash
conda create -n EMusicGen python=3.9 --yes
conda activate EMusicGen
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Maintenance
```bash
git clone git@gitee.com:MuGeminorum/EMusicGen.git
cd EMusicGen
```

## Success rate
| Dataset | Rough4Q | VGMIDI | EMOPIA |
| :-----: | :-----: | :----: | :----: |
|    %    |   99    |   93   |   27   |

## Experiments
|   Control   |   Q1    |   Q2    |  Q3   |  Q4   |
| :---------: | :-----: | :-----: | :---: | :---: |
| Tempo (BPM) | 160-184 | 184-228 | 40-69 | 40-69 |
|    Mode     |  major  |  minor  | minor | major |
|    Pitch    |    0    |   -24   |  -12  |   0   |
| Volume (dB) |   +5    |   +10   |   0   |   0   |

## Performance
In different control modes, generate music segments using specified emotional prompts. Have three groups of people label these music segments in a blind listening test. Compare the overlap between the prompts and the labels to describe performance.

| Ablation | Accuracy | F1-score |      Confusion matrix      |
| :------: | :------: | :------: | :------------------------: |
|  Tempo   |  0.660   |  0.649   | ![](./figs/mat-tempo.jpg)  |
| Pitch SD |  0.670   |  0.648   |  ![](./figs/mat-std.jpg)   |
|   Mode   |  0.710   |  0.708   |  ![](./figs/mat-mode.jpg)  |
|  Octave  |  0.720   |  0.712   | ![](./figs/mat-pitch.jpg)  |
|  Volume  |  0.860   |  0.859   | ![](./figs/mat-volume.jpg) |
|   None   |  0.910   |  0.909   |  ![](./figs/mat-none.jpg)  |

## Future work
Referring to the RLBH of InstructGPT, we will introduce a PPO reinforcement learning fine-tuning optimization for the tunesformer model as well.

![](./figs/ppo.png)

## Cite
```bibtex
@article{Zhou2024EMusicGen,
  title     = {EMusicGen: Emotion-Conditioned Melody Generation in ABC Notation},
  author    = {Monan Zhou, Xiaobing Li, Feng Yu and Wei Li},
  month     = {Sep},
  year      = {2024},
  publisher = {GitHub},
  version   = {0.1},
  url       = {https://github.com/monetjoe/EMusicGen}
}
```

## Thanks
Thanks to [@sanderwood](https://github.com/sanderwood) and [@ElectricAlexis](https://github.com/ElectricAlexis) for providing technical supports on data processing and feature extraction.