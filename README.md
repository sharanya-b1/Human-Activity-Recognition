## Human Activity Classification

### Steps to Execute
1) Download the x_test binary file and put in in the same location as the python script. 
(https://drive.google.com/file/d/1cO-7hHGZecQxYXAzPPdQYMUtmOegx35I/view?usp=share_link)

2) The final directory structure should look like this
Team_12
|
|------ ``script.py``
|
|------models
|
|------``y_test``
|
|------``x_test``
|
|------``README.md``
|
|------``requirements.txt``
|
|------training

3) The models folder contains the models (SVM, decision trees, logistic regression and neural network). The training folder contains the preprocessing and training code in the form of ipynb files. ``script.py`` is the inference code. Use ``python3 script.py`` to run the code.
4) Select a model and select the hyperparameter(if asked) and the script would give the classification report containing the accuracy f1-score, etc generated from x_test and y_test.

### Description of codebase
1) The first step is to preprocess the dataset. The NaN values have been linearly interpolated from nearby valid values.

2) First logistic regression has been used to find a baseline for the model. Different values of C have been tested for this.

3) For SVM first a grid search is performed on a small part of the dataset to find optimal hyperparameters like kernel type and C value. Then SVM the SVM has been trained on a slightly larger part of the dataset. The entire dataset cant be processed using SVM as it is too big. Linear kernel with C=0.1 has been taken

4) For neural network model, a dropout of 0.2 has been taken as compared to 0.5 in the paper. The model has been trained for 15 epochs. A slightly  higher accuracy might be obtained if trained for more number of epochs.

5) For ordinary decision tree, max depth of 15 with minimum sample split of 10 has been used.

6) For boosted decision trees, max depth of 3 with earning rate of 0.05 has been trained for 600 iterations.

7) For random forest, a max depth of 3 was used with 300 estimators.

8) The above methods are repeated for the partial feature dataset containing data only from the hand IMU and the heartrate.



### Misc.
1) The ``script.py`` file only runs on the full feature dataset but the partial feature models have been included in the submission.

2) The training folder contains the notebooks for video classification on UCF and other models which we have explored.

3) ``requirements.txt`` contains the requirements for only the main project (on the PAMAP2 dataset)

4) The PAMAP2 dataset can be downloaded at - https://drive.google.com/drive/folders/114oSmTfKEnTmaXk9hdVPMdXs1KbZIrxY?usp=share_link

5) The UCF50 dataset can be downloaded at - https://drive.google.com/drive/folders/1V4QvWDL1xv99dYymLFrqmkumdE0PEW17?usp=share_link

6) The HAR dataset can be downloaded at - https://iiitaphyd-my.sharepoint.com/personal/sharanya_b_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsharanya%5Fb%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2FSMAI%2DS23%2DProjectFiles%2FUCI%20HAR%20Dataset%2Ezip&parent=%2Fpersonal%2Fsharanya%5Fb%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2FSMAI%2DS23%2DProjectFiles&ga=1

7) The HMDB51 dataset can be downloaded at - https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

8) The ``Extra Experiments`` in the training folder contains the python notebooks for the other experiments we have performed.

9) The SVM model has not been included in ``script.py`` as it requires extra preprocessing steps. It can be viewed in the ipynb file in the training folder.


