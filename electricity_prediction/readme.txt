=============================
Code folder
=============================
main.py:
 - locates main()
parameters.py
 - stores all options, parameters and paths to data and other folders
econ_data.py
 - From an original source file, extract the user named 250 and save it to a cvs file
input_data.py
 - processes the input file and makes train and test data set
econ_model.py
 - creates model and has train(), evaluate() and predict()
visualization.py
 - displays charts of losses and prediction
eda_econ.ipynb
 - is similar to econ_data but notebook code

=============================
data folder
=============================
LD2011_2014.txt
 - an original file downloaded from UCI machine learning repository
LD_250.npy
 - is extracted from LD2011_2014.txt only for user “250”
data_with_date_250
 - is data of 250 with date

=============================
data folder
=============================
LD_250.npy
 - electric spending for 250 user
LD2011_2014.txt
 - all electric data
data_with_date_250.csv
 - data of 250 with date

=============================
result folder
=============================
result.csv
 - recent prediction result
experiment_log
 - raw experimental data
result_with_date
 - result with date for reference

=============================
ckpt folder
=============================
stateless folder
stageful folder
- all placeholders’ folder

=============================
practice folder
=============================
chart_datetime.ipynb
 - example code to display date on X axis

