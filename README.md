# EMusicGen
[![Python application](https://github.com/monetjoe/EMusicGen/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/EMusicGen/actions/workflows/python-app.yml)

Emotionally Conditioned Music Generation in ABC Notation with Chord

![](https://github.com/monetjoe/EMusicGen/assets/20459298/9ee364d5-f80f-460d-9154-58b85ad59d15)

Referring to the RLBH of InstructGPT, we introduce a PPO reinforcement learning fine-tuning optimization for the tunesformer model as well.

## Requirements
```bash
conda create -n gpt python=3.8 --yes
conda activate gpt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Maintenance
```bash
git clone git@gitee.com:MuGeminorum/EMusicGen.git
cd EMusicGen
```

## Success rate
| Dataset |   %   |
| :-----: | :---: |
| Rough4Q |  99   |
| EMOPIA  |  27   |
| VGMIDI  |   -   |
