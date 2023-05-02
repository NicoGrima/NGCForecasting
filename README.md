# NGCForecasting

### Libraries:
Install any missing dependencies by running the following command in your terminal:

```pip install -r requirements.txt```

### Loading the data:
Make sure that you load either file that has no missing values: "9636_10.sqlite" or "8101_31.sqlite". Make sure that the filename (variable: 'data_path' in 'main.py') is named appropriately.

### Selecting model:
Select either 'LSTM' or 'Transformer' in the 'model_select' variable in 'main.py'.

### Selecting evaluation method:
Select whether to use k-folds cross validation (toggle 'cross_val' to either 'True' or 'False' in 'main.py').

### Tuning models:
Hyperparameters are defined in the 'Define the model' section within 'main.py'. Also epochs, batch size, and input and output lengths can be found in the 'Define parameters' section in 'main.py'. We can tune the models by changing these variables' values.
