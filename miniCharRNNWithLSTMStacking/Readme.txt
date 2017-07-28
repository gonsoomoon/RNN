	
1.Project 
 - "Character-Level Language Model(CLLM)” through stacking LSTM in Tensorflow”
2.Description
 - CLLM is implemented in Tensorflow and compare a change of loss by a range of LSTM stacking level such as 1, 2 and 3
3.Date
 - July 2017
4.Engineer
 - Gonsoo Moon
5.Environment
 1) Mac Pro 2.6 GHz Intel Core i5, 16 GB RAM
 2) Tensorflow 1.0.0
 3) Python 2.7

6.How to run
 On the command line, type the following
 python2 rnn_long_char_master.py 'data/simple.txt' 1 10 10 100
 python2 rnn_long_char_master.py 'data/simple.txt' 2 10 10 100
 python2 rnn_long_char_master.py 'data/simple.txt' 3 10 10 100
 Note:
 # 1: LSTM Level
 # 10: sequence length
 # 10: moving window. in other words, exactly split a text into a batch.
 # 100: size of epoch

7.Result
 - Refer to the project note
 - The source code made a event log for tensorboard so that you can launch TensorBoard 
  with the following command:
  tensorboard --logdit = './logs'
  and Open 127.0.0.1:6006 on the web browser
	   


	
