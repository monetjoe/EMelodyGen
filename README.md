# mids2abc_dataset
[![Python application](https://github.com/monet-joe/mids2abc_dataset/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monet-joe/mids2abc_dataset/actions/workflows/python-app.yml)

Convert MIDI into abc jsonl dataset

![绘图1](https://github.com/monet-joe/mids2abc_dataset/assets/20459298/9484c3a4-1c5b-4248-ad43-e52012dcddac)

## Maintenance
```bash
git clone git@gitee.com:MuGeminorum/mids2abc_dataset.git
cd mids2abc_dataset
```

## Environment
```bash
conda create -n data --yes --file conda.txt
conda activate data
pip install -r requirements.txt
```

## Usage
Put midis into `./data/mids` and run:
```bash
python processor.py
```

## XML Slicer
According to the statistics on the Irishman dataset, the average number of bars for all XML scores in its validation set is `19.824699352451432`. Therefore, we set the cut length of the XML slicer to 20 bars, and most of the end slices of each tune are generally smaller than 20 bars, which exactly results in the average number of bars for the overall sliced data slightly lower than 20, consistent with the Irishman dataset.

Note that the average number of bars in the XML here is the value calculated without expanding the repeated bars, we later tried to convert all the XML in the validation set of the Irishman dataset to midi (i.e., after expanding the repeated bars) and recalculated it, and the result was about 32 bars after taking the repeated bars into account.

However, since the ABC score and the XML score data are of equal status, the shorter the slice length is, the less likely to cause training distortion, and after cutting at the midi level, we still need to convert back to XML to ensure that the conversion to abc contains normal structural information, and multiple conversions are likely to result in the loss of details and distortion of the data, therefore, the average number of bars of the midi mentioned above is only for reference, and we still choose to 20 bars as the slicing unit of the XML score as the slicing unit of XML.

## Transpose ABC Notations
transpose abc scores to 15 tones:

![image](https://github.com/monet-joe/abc_transposition/assets/20459298/776fc0cd-6f48-4c68-90aa-084915252e05)

## Test WebUI
```bash
python app.py
```
Then operate on <http://127.0.0.1:7860> or http://localhost:7860, and check logs in the terminal.
