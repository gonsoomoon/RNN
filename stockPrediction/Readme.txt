	
1.Project 
 - Stock prediction using LSTM(Long Short-Term Memory)
2.Description
 - With a long historical stock data about 10 years and LSTM RNN(Recurrent Neural Network), a specific stock price is predicted. 
3.Date
 - Aug 2017
4.Engineer
 - Gonsoo Moon
5.Environment
 1) Mac Pro 2.6 GHz Intel Core i5, 16 GB RAM
 2) Tensorflow 1.0.0
 3) Python 3.6

6.How to run
 On the command line, type the following
 python rnn_stock_prediction_master.py 'data/google.csv' 'result/google-1200.csv' 'result/google-loss-1200.csv' 0.8 3000 1200
 Note:
 # ‘data/google.csv’: the input data file
 # ‘result/google-1200.csv: Target, prediction and difference are stored
 # ‘result/google-loss-1200.csv’: a value of loss is stored
 # 0.8: A ratio of train to test data
 # 3000: # of sequence length of LSTM (—> Please use a small size at first like 20)
 # 1200: # of epochs (—> Please use a small size at first like 40)

7.Result
 - Refer to the project report

	   


	
