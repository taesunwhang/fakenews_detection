### Fake News Detection
Implementation of BERT baseline for fakenews detection (fnc-1)

### Requirements
- Code is written in Python (3.6) and requires tensorflo-gpu-1.13.1
- This code is implemented based on https://github.com/google-research/bert

### Data Preprocessing
The dataset is publicly avaliable in https://github.com/FakeNewsChallenge/fnc-1

To make dataset for training BERT, run

    python3 fnc-1/data_utils.py

### Training BERT for fake news detection (evaluated on fnc-1 dataset)
    python3 main.py --model=bert_base

### Esamples of training output
    [Step 100][100 th] loss: 4.8840, accuracy: 67.69%  (86.73 seconds)
    [Step 200][200 th] loss: 4.0117, accuracy: 83.12%  (78.43 seconds)
    [Step 300][300 th] loss: 3.6083, accuracy: 85.50%  (78.28 seconds)
    [Step 400][400 th] loss: 3.5423, accuracy: 85.56%  (79.31 seconds)
    [Step 500][500 th] loss: 2.8391, accuracy: 87.94%  (78.33 seconds)