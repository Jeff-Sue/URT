# Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation

This repository is the official implementation of "Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation".

## Latest News
- [4/7/2024]: Release the new version. The new image and results will update as soon as possible.
- [3/6/2024]: Release the code of the paper.
- [30/5/2024]: Release the [paper](https://arxiv.org/abs/2405.19799).


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

NSP-BERT: This model was trained following the DialSTART approach, with modifications to the structure and input of the cls layer. The resulting model serves as a pre-trained model for topic modeling, yielding a basic topic matrix. You can download it from the [link](https://drive.google.com/file/d/12BzNwtbMyTL2jaEKpXOTWeOeqRJ55orF/view?usp=drive_link).


## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python main.py
```

## Results

Our model achieves the following performance on :


| Model name         | STAC(F1↑) |Molweni(F1↑) | Doc2Dial(Pk↓, WD↓) | TIAGE(Pk↓, WD↓)|
| ------------------ |---------------- | -------------- |---------------- | -------------- |
| DialSTART  |     -    |     -     | 45.92    50.25 | 50.28    54.23|
| Bart-Large  |     37.00    |     42.25     | -    - | -    -|
| Simple Addition  |     34.79    |     42.67     | 47.75    52.23 | 49.70    53.12|
| Ours-Full Dataset  |     38.38     |     46.24      | **44.96    49.49** | 48.53    **53.00** |
| Ours-STAC  |     39.02     |     42.67     | 45.75    50.25 | 49.88    54.41 |
| Ours-Molweni  |     **40.22**     |     **50.55**    | 51.36    58.22 | 48.72    55.89 |
| Ours-Doc2Dial  |     37.09     |     44.94    | 48.28    52.73 | **46.85**    53.54 |
| Ours-TIAGE  |     39.21   |     46.07    | 45.04    49.60 | 48.84    53.17 |

## Citation

```
@misc{xu2024unsupervised,
      title={Unsupervised Mutual Learning of Dialogue Discourse Parsing and Topic Segmentation}, 
      author={Jiahui Xu and Feng Jiang and Anningzhe Gao and Haizhou Li},
      year={2024},
      eprint={2405.19799},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
