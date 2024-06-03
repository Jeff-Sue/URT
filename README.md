# Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation

This repository is the official implementation of "Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation".

## Introduction
In this paper, we propose an unsupervised joint modeling of rhetoric and topic structures under the discourse in linguistics, enabling mutual learning between the two.
![Model Architecture](https://github.com/Jeff-Sue/URT/blob/main/main2.png)

## Motivation
- In order to address the issue that current large models are unable to simultaneously meet the needs of third-party companies and users. We believe that the joint modeling of topic and rhetorical structures can help in understanding and controlling the direction of the entire conversation.
- In linguistics, topics and rhetorical structures are often interrelated and influential, but previous work has focused only on one aspect or used one to assist the other. We have decided to be the first to attempt a joint modeling of both.
- Due to the issues of data sparsity and information scarcity in both rhetoric and topics, we have chosen to conduct joint modeling in an unsupervised setting.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

Here, we give the sources of four dataset we used, please download them and put in the **data/{dataset_name}/** files:

**STAC**: https://github.com/chijames/structured_dialogue_discourse_parsing/tree/master/data/stac

**Molweni**: https://github.com/chijames/structured_dialogue_discourse_parsing/tree/master/data/molweni

**Doc2Dial**: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/dial-start

**TIAGE**: https://github.com/HuiyuanXie/tiage

## Pre-trained Models

Source of pre-trained models:

Bart-Large-CNN: https://huggingface.co/facebook/bart-large-cnn

NSP-BERT: This model was trained following the DialSTART approach, with modifications to the structure and input of the cls layer. The resulting model serves as a pre-trained model for topic modeling, yielding a basic topic matrix. You can download it from the link: https://drive.google.com/file/d/12BzNwtbMyTL2jaEKpXOTWeOeqRJ55orF/view?usp=drive_link


## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python main.py
```

## Results

Our model achieves the following performance on :

<style>
    .highlight {
        background-color: yellow;
    }
</style>

| Model name         | STAC(F1) |Molweni(F1) | Doc2Dial(Pk, WD) | TIAGE(Pk, WD)|
| ------------------ |---------------- | -------------- |---------------- | -------------- |
| My awesome model   |     38.38     |     46.24      | 44.96 49.49 | 48.53 53.00 |{: .highlight }
