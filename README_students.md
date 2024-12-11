### External_data_sorted

This file purpose was to be used as a testbed to fine tune our approach to features engineering in the external_data set.

The focus was first set on conducting operations on the dataset himself. The aim was to quantify the presence of NaN values, outliers and overall quality and relevance of the data.

In order to "fill" the gap, as the external_data set is only sampled once every three hours, two lines had to be created. Experience showed creating lines on the model (copy-2h / copy-1h / actual_data-h) to be the most effective method.

Once these sortings have been carried, various models have been tested. XGBoost eventually prevailed as the most suitable for this experiment. It was iteratively enhanced through GridSearchCV, then Optuna.

One notable point is the presence of numerous NaN in the external_data, including within the columns that were selected for this model. Experience proved  neither inputations methods nor removing these lines altogether to be effective. XGBoost embedded ability to sort and deal with NaN values has proven more effective than any solutions we came up with.

Moreover, additionnal external datasets have been tested, including but not limited to : strikes, pollution level, road blocks and construction sites... however the results of which have been disappointing.

The final training dataset is comprised of the following columns :

dtypes : DateTime

date

dtypes : float

etat_sol (state of the ground)
ff (wind speed)
t (temperature)
u (humidity)
vv (visibility)
n (clouds)
ht_neige (snow height)
rr3 (rainfall over the last three hours)

dtypes : bool

East
South
West
Lockdown
soft-curfew
hard-curfew
bank_holidays (named "is_holidays" in the actual model)
holidays (named "vacations" in the actual model)

### External_data_sorted

The actual model runs directly by pressing play in the main.py file. It is comprised of four distinct models that work in conjunction :

- main.py, which gives inputs on which dataset to use, how to access it and which data to predict
- data_processing.py, which cleans, sorts and enhance the external_data.csv and merges it with the train.parquet
- model.py, which is used to set models parameters and outputs
- eval.py, which turns the output into a readable csv
