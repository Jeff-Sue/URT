# Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation

This repository is the official implementation of "Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation".

## Introduction
In this paper, we propose an unsupervised joint modeling of rhetoric and topic structures under the discourse in linguistics, enabling mutual learning between the two.
![Model Architecture]()
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

Source of the data:

STAC: https://github.com/chijames/structured_dialogue_discourse_parsing/tree/master/data/stac

Molweni: https://github.com/chijames/structured_dialogue_discourse_parsing/tree/master/data/molweni

Doc2Dial: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/dial-start

TIAGE: https://github.com/HuiyuanXie/tiage

## Pre-trained Models

Source of pre-trained models:

Bart-Large-CNN: https://huggingface.co/facebook/bart-large-cnn

NSP-BERT: This model was trained following the DialSTART approach, with modifications to the structure and input of the cls layer. The resulting model serves as a pre-trained model for topic modeling, yielding a basic topic matrix.


## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python main.py
```

## Results

Our model achieves the following performance on :

| Model name         | STAC(F1) |Molweni(F1) | Doc2Dial(Pk, WD) | TIAGE(Pk, WD)|
| ------------------ |---------------- | -------------- |---------------- | -------------- |
| My awesome model   |     38.38     |     46.24      | 44.96 49.49 | 48.53 53.00 |
