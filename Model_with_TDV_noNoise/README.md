
The total dataset is not included in this repo due to the size. If you would like to use it, please contact the author.

The trained model **model_para_4.pt** is already included, simply run **test_for_paper.py** (modify the data path) with the dataset to see the results on the testing set.

For the prediction, there are 3 Kepler 88 TTVs (see last functions in **train_utils.py**), one is based on the observation (Holczer et al., 2016) with missing transits, the other is interpolated from the previous one, and the last one is obtained by running GRIT assuming parameters are those from Nesvorny et al., 2013.


### 1. train.py
1.1 Add your data path in **train.py**.
1.2 If you would like to do grid search, you may change Hyper Parameters in **train.py**
1.3 Long training epoches, e.g., 200, could need to resume the training under certain circumstances. More information is included in the code (search **Continue_train**).
1.4 run **train.py** to start training. You may use **tensorboard** to check the training, data is included in folder **runs_200**

### 2. models/gru_vis_trans.py
This file includes the model architecture.

### 3. performance_test_set.py
This file is for predicting 3 different systems, i.e., Table 4 in the paper. 

### 4. test_for_paper.py
This file is to output the results on the testing set.


