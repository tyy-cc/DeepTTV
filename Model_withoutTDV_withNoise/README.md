
The total dataset is not included in this repo due to the size. If you would like to use it, please contact the author.

The trained model **model_para.pt** is already included, you may reuse and modify **test_for_paper.py** from other model folder to see the results on the testing set.

For the prediction, there are 3 Kepler 88 TTVs (see last functions in **train_utils.py**), one is based on the observation (Holczer et al., 2016) with missing transits, the other is interpolated from the previous one, and the last one is obtained by running GRIT assuming parameters are those from Nesvorny et al., 2013.

### 1. train.py
1.1 Add your data path in **train.py**.
1.2 If you would like to do grid search, you may change Hyper Parameters in **train.py**
1.3 run **train.py** to start training. You may use **tensorboard** to check the training, data is included in folder **runs**

### 2. models/gru_vis_trans.py
This file includes the model architecture.

### 3. test.output
This file is the output of **train.py**.
