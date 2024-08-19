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
| Rough4Q |  100  |
| VGMIDI  |  93   |
| EMOPIA  |  27   |

## Performance
In different control modes, generate music segments using specified emotional prompts. Have three groups of people label these music segments in a blind listening test. Compare the overlap between the prompts and the labels to describe performance.

| Control | Accuracy | F1-score |     Confusion matrix      |
| :-----: | :------: | :------: | :-----------------------: |
|   All   |  0.820   |  0.816   |  ![](./exps/mat-all.jpg)  |
|  Pitch  |  0.690   |  0.679   | ![](./exps/mat-pitch.jpg) |
|  Mode   |  0.650   |  0.654   | ![](./exps/mat-mode.jpg)  |
|  Tempo  |  0.530   |  0.487   | ![](./exps/mat-tempo.jpg) |
|  None   |  0.520   |  0.498   | ![](./exps/mat-none.jpg)  |