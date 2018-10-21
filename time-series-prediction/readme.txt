The following explains code and resource files

=============================
electricity folder
=============================
multi/e235_multi_feature_v01.ipynb
- with two features, making a model on two lstm layers
multi/e235_e250_e252_multi_feature_v01.ipynb
- with two features, making a model on three people
multi/e235_e250_e252_multi_feature_v02.ipynb
- adding more test data after loading a pre-trained model
multi/inference_multi_feature_v01.ipynb
- inference function from the model that the e235_multi_feature_v01.ipynb creates
single_feature/e235_lstm_v01.ipynb
- with 24 lags and 24 presses on two lstm layers
single_feature/e235_e250_e252_lstm_v01.ipynb
- On three people data, making a model
single_feature/24_lags_24_presteps_2_lstm_infer_v01.ipynb
- inference from the model that e235_lstm_v01.ipynb creates
multiple_lags_multiple_presteps_2_lstm_v0.1.ipynb
- Two stacked LSTM with a class of parameters
multiple_lags_1_presteps_lstm_v0.1.ipynb
- stageful LSTM with multiple lags and one-step prediction on electricity spending
electricity_input.ipynb
- From an original file, extract each person’s electric spending


=============================
practice folder
=============================
tutorial_add_numbers.ipynb
- adding two numbers on seq2seq
tutorial_multvariate_lstm.ipynb
- multivariate features with last
tutorial_category_embedding_v0.1.ipynb
- several ways of category embedding
tutorial_multi-step_time_LSTM.ipynb
- multi-step prediction on shampoo data set
Intro to autocorrelation.ipynb
- explains what autocorrelation and partial autocorrelation
time_series_to_supervised_learning_problem.ipynb
- handles converting time-series data to supervised problem
intro_acf_pacf.ipynb
- handles acf() and pacf()
tutorial_datetime_timezone.ipynb
- loads date time string, change to a time zone and aggregate 
tutorial_basic_time_series_visualization.ipynb
- shows basic time-series visualization
tutorial_advanced_time_series_visualization.ipynb
- handles resampling, asfreq of samsung security, grouping by weekend and hour
in_depth_linear_regression.ipynb
- includes linear regression, regularization (L1, L2), bootsrap resampling
baseline-shampoo_sales.ipynb
- makes a persistence model
ARIMA_shampoo_sale.ipynbs
- makes a basic ARIMA model on shampoo sale data
preprocess-hour-series-feature-bicycle.ipynb
- make time-series feature set for supervised learning
tutorial_time_series_neural_network.ipynb
- has NN, LSTM implementation on airport passenger data file
tutorial-mlp-shampoo_sales.ipynb
- experiment time-series on multiple perceptron network varing # of neurons and epochs


=============================
input folder
=============================
electricity
- three people’s electric spendings: elect_250.csv, elect_235.csv, elect_37.csv
- three people’s electric spendings with person id: elect_250w_id.csv, elect_235w_id.csv, elect_37w_id.csv
samsung_20180608.csv
- Samsung security
FremontBridge.csv
- counts of borrowing bicycle in Seattle
sales-of-shampoo-over-a-three-ye.csv
- shampoo sales data
daily-minimum-temperatures-in-me.csv
- Australia temperature data
timeseries.txt
- data file for the tutorial_datetime_timezone.ipynb in the practice folder
seattle_weather.csv
- seattle temperature, precipitation and wind strength
airport-passenger
- airport passenger data file

=============================
util folder
=============================
fresh_general_util.py
- utility library
fresh_input_preprocess.py
- input-related library
fresh_prediction.py
- prediction-related library
fresh_parameters.py
- has parameter class
fresh_model.py
- model library

general_util.py
- utility library
input_preprocess.py
- input-related library
prediction.py
- prediction-related library
ts_input.py
 - has an input() for time-series data
util_test.py
 - tests files in the util folder

=============================
result folder
=============================
experiment_log_v0.3
- experiment log
experiment_log_v0.2
- experiment log
