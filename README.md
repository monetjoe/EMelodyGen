# EMusicGen
[![license](https://img.shields.io/badge/License-LGPL-yellow.svg)](https://github.com/monetjoe/EMusicGen/blob/master/LICENSE)
[![Python application](https://github.com/monetjoe/EMusicGen/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/EMusicGen/actions/workflows/python-app.yml)
[![ds](https://img.shields.io/badge/ModelScope-Dataset-624aff.svg)](https://www.modelscope.cn/datasets/monetjoe/EMusicGen)
[![model](https://img.shields.io/badge/ModelScope-Model-624aff.svg)](https://www.modelscope.cn/models/monetjoe/EMusicGen)
[![demo](https://img.shields.io/badge/ModelScope-Demo-624aff.svg)](https://www.modelscope.cn/studios/monetjoe/EMusicGen)
[![ds](https://img.shields.io/badge/HuggingFace-Dataset-ffd21e.svg)](https://huggingface.co/datasets/monetjoe/EMusicGen)
[![model](https://img.shields.io/badge/HuggingFace-Model-ffd21e.svg)](https://huggingface.co/monetjoe/EMusicGen)
[![demo](https://img.shields.io/badge/HuggingFace-Demo-ffd21e.svg)](https://huggingface.co/spaces/monetjoe/EMusicGen)

Emotionally Conditioned Melody Generation in ABC Notation

![](./figs/model.jpg)

## Environment
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## Maintenance
```bash
git clone git@github.com:monetjoe/EMusicGen.git
cd EMusicGen
```

## Train
```bash
python train.py
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
However, our current work still faces several limitations. For instance, conclusions derived from statistical correlations only provide a rough guide for designing emotional templates and do not fully reflect the true distribution of features within the emotional space. Additionally, due to the relatively small amount of data and the concentration on pop and game music styles, our analysis results are susceptible to [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox). Furthermore, melody generation based on emotional control templates often results in music that is concentrated on a few specified emotions, rather than representing the complete emotional quadrant. For instance, when aiming to generate music for the Q2 quadrant, specifying templates may lead to a concentration on tense music, whereas anger and some other emotions also fall within Q2. Although this approach allows for high precision in 4Q representation, it may lead to a lack of emotional diversity in the generated music.

To address these issues, we have released an application demonstration on [HuggingFace](https://huggingface.co/spaces/monetjoe/EMusicGen) based on the inference code of our generation system. This demonstration enables users to design and specify emotional templates, utilizing large-scale data to progressively refine feature distributions for greater accuracy. Additionally, the error-free rate is merely a necessary condition for quality but does not fully reflect the true quality of the generated melodies. Future work could incorporate reinforcement learning feedback in the demonstration to adjust the system's generation quality based on user-generated evaluations. Furthermore, while this study focuses on melody, chords are a crucial factor influencing musical emotion. Therefore, our demonstration also includes an option to add chords, and their impact will be considered in future research.

![](./figs/ppo.png)

Referring to the RLBH of InstructGPT, we will introduce a PPO reinforcement learning fine-tuning optimization for the tunesformer model as well.

## Cite
```bibtex
@inproceedings{Zhou2024EMusicGen,
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