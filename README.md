# HTF-TP

The implemention of paper "Trajectory Prediction with Heterogeneous Graph Neural Network" [paper](https://github.com/Alue111/files/blob/main/Trajectory%20Prediction%20With%20Heterogeneous%20Graph%20Neural%20Network.pdf)


### Processing data
Please donwload the raw datasets and run the following command to process data
'''
python utils/data_process.py
'''

### Training
Run the following command to train our model
'''
python scripts/training_loop.py
'''

### Evaluation
Run the following command to evaluate the performance:
'''
python scripts/test_pretrained_model.py
'''


