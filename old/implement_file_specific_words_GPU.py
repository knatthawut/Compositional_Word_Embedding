# -*- coding: utf-8 -*-
import functools
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sets
import scipy.spatial
from gensim.models import Word2Vec
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pprint
pp = pprint.PrettyPrinter(indent=4)

import function_files
import json
from scipy import stats
import time
# *************************************************************************************************************************
# 
# Setting Environment & Functions
# 
# *************************************************************************************************************************

learning_rate = 0.00023
training_iters = 5000
batch_size = 50000

KEEP_PROB = 0.68

#Editing
n_input = int(sys.argv[1])
fold_num = int(sys.argv[2])
type_model = sys.argv[3]

num_hidden = 128

n_classes = n_input # classfication type (y/n)
max_length = 21
num_cnt = 2

itr_notChange = 10
threshold_loss = 0.000001

filename = 'uni_pair_combine_less10_'
vectorFile = './'+type_model+'/wiki-db_more50_'+str(n_input)+''# sys.argv[1]

# loadpath,savepath,finalresultfile
loadPath = "./model/"+str(n_input)+"/"+filename+'_'+str(fold_num)+".ckpt"
loadPath_o = "./model/"+str(n_input)+"/"+filename+'_'+str(fold_num)+"_original.ckpt"
loadFlag = False
savePath = "./model/"+str(n_input)+"/"+filename+'_'+str(fold_num)+".ckpt"
savePath_o = "./model/"+str(n_input)+"/"+filename+'_'+str(fold_num)+"_original.ckpt"
finalResultFile = str(n_input)+'_'+type_model+'.result'
finalResultFile_o = str(n_input)+'_'+type_model+'_original.result'

filenameData = 'uni_pair_combine_less10_'

# *****************************************************************************************************************
# 
# load Word Vector & Method
# 
# *****************************************************************************************************************
wordvec = Word2Vec.load(vectorFile)
wordvec.init_sims(replace=True)
model_item = wordvec.vocab.items()
print('Loading W2V Model Completed')


# return the length of data
def _length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	length = tf.reduce_sum(used, reduction_indices=1)
	length = tf.cast(length, tf.int32)
	return length
	
def loadData(train_data,model,start,bs):
	train = []
	label = []
	not_found = 0
	counter = 0
	no_len = 0
	for line in train_data[start:start+bs]:
		data = line[:-1].split('\t')
		counter = counter + 1
		if len(data) == 2:		
			if data[0] in model:
				label.append(model[data[0]])
				train.append(np.array(function_files.getSample(data[1].split(),model), dtype = np.float).astype(np.float32))   
			else:
				#print 'Not Found :',data[0]
				not_found = not_found + 1
		else:
			print('Length Error')  
			no_len = no_len + 1
	#print 'Count',str(counter),'Not found',str(not_found),str(no_len)
	return np.array(train, dtype = np.float).astype(np.float32), np.array(label, dtype = np.float).astype(np.float32)	

def loadDataTest(line,model):
	train = []
	label = []
	not_found = 0
	counter = 0
	no_len = 0
	data = line[:-1].split('\t')
	counter = counter + 1
	if len(data) == 2:		
		if data[0] in model:
			label.append(model[data[0]])
			train.append(np.array(function_files.getSample(data[1].split(),model), dtype = np.float).astype(np.float32))   
		else:
			#print 'Not Found :',data[0]
			not_found = not_found + 1
	else:
		print('Length Error')	
		no_len = no_len + 1
	return np.array(train, dtype = np.float).astype(np.float32),np.array(function_files.avgSample(data[1].split(),model),    dtype = np.float).astype(np.float32),    np.array(label, dtype = np.float).astype(np.float32)	

def saveModel(sess,path):
	save_path = saver.save(sess,path)

def saveModel_o(sess,path):
	save_path = saver.save(sess,path)



# *****************************************************************************************************************
# 
# Setting Tensorflow Structure For RNN with Attention
# 
# *****************************************************************************************************************
attention_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_attention = tf.Session(graph = attention_graph,config=tf.ConfigProto(gpu_options=gpu_options))

num_hidden = 128
num_layers = 2

with attention_graph.as_default():
	# *********************
	# Set placeholder & RNN
	# *********************
		data = tf.placeholder(tf.float32, [None, max_length, n_input])
		target = tf.placeholder(tf.float32, [None, n_classes])
		keep_prob_ph = tf.placeholder(tf.float32)

		# Setting Recurrent Network & Getting the last output
		output, last = rnn.dynamic_rnn(
			rnn_cell.GRUCell(num_hidden),
			data,
			dtype=tf.float32,
			sequence_length=_length(data),
			time_major=False
		)
		# (?, 21, 128),(?, 128)

	# ***************
	# Attention Layer
	# ***************
		attention_size=128

		output_shape = output.get_shape()
		sequence_length = output_shape[1].value  # the length of sequences processed in the antecedent RNN layer
		hidden_size = output_shape[2].value  # hidden size of the RNN layer
			    
		# Attention mechanism
		W_omega = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=1.0))
		b_omega = tf.Variable(tf.truncated_normal([attention_size], stddev=1.0))
		u_omega = tf.Variable(tf.truncated_normal([attention_size], stddev=1.0))

		v = tf.nn.relu(tf.matmul(tf.reshape(output, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
		vu = tf.matmul(  v, tf.reshape(u_omega, [-1, 1])  )
		exps = tf.reshape(   tf.exp(vu), [-1, sequence_length]  )

		# (?, 21)
		alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

		# Output of Bi-RNN is reduced with attention vector (?, 128) =(?, 21, 128) * (?, 21, 1)
		attention_output = tf.reduce_sum(output * tf.reshape(alphas, [-1, sequence_length, 1]), 1)


		# ********
		# Drop out (?, 128)
		# ********
		drop_output = tf.nn.dropout( attention_output, keep_prob_ph )
		# drop_output = attention_output


		in_size = num_hidden
		out_size = int(target.get_shape()[1])
		weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
		bias = tf.constant(0.1, shape=[out_size])
		weight = tf.Variable(weight)
		bias = tf.Variable(bias)

		prediction = tf.matmul(drop_output, weight) + bias

	# *****************************************
	# Setting Loss and Optimizer & Initializing
	# *****************************************
		loss = tf.reduce_mean(  tf.square(target - prediction)  )
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver() 	


tf.reset_default_graph()
original_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_original = tf.Session(graph = original_graph,config=tf.ConfigProto(gpu_options=gpu_options))

with original_graph.as_default():
	# ***************
	# Set placeholder
	# ***************
		data_o = tf.placeholder(tf.float32, [None, max_length, n_input])
		target_o = tf.placeholder(tf.float32, [None, n_classes])

	# ******************
	# Recurrent network.
	# ******************
		output_o, last_o = rnn.dynamic_rnn(
			rnn_cell.GRUCell(num_hidden),
			data_o,
			dtype=tf.float32,
			sequence_length=_length(data_o),
			time_major=False
		)

		in_size_o = num_hidden
		out_size_o = int(target_o.get_shape()[1])
		weight_o = tf.truncated_normal([in_size_o, out_size_o], stddev=0.01)
		bias_o = tf.constant(0.1, shape=[out_size_o])
		weight_o = tf.Variable(weight_o)
		bias_o = tf.Variable(bias_o)

		prediction_o = tf.matmul(last_o, weight_o) + bias_o

	# *****************************************
	# Setting Loss and Optimizer & Initializing
	# *****************************************
		loss_o = tf.reduce_mean(  tf.square(target_o - prediction_o)  )
		optimizer_o = tf.train.AdamOptimizer(learning_rate).minimize(loss_o)
		init_o = tf.global_variables_initializer()
		saver_o = tf.train.Saver() 	

tf.reset_default_graph()

calculate_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_calculate = tf.Session(graph = calculate_graph,config=tf.ConfigProto(gpu_options=gpu_options))

with calculate_graph.as_default():
	rnn = tf.placeholder(tf.float32, [None, 400])
	ref = tf.placeholder(tf.float32, [None, 400])

	distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(rnn, ref)))))

	normalize_a = tf.nn.l2_normalize(rnn,0)        
	normalize_b = tf.nn.l2_normalize(ref,0)
	cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))

	init_cal = tf.global_variables_initializer()

# ******************************************************************************************************************************
# 
# Loading Training and Testing Data
# 
# ******************************************************************************************************************************
t_test_value = 0.0
t_test_value_o = 0.0

avg_p_value = 0.0

avg_last_epoch_error = 0.0
avg_eucliden_acc = 0.0
avg_cosine_acc = 0.0
avg_MRR = 0.0
avg_hit = 0

avg_last_epoch_error_o = 0.0
avg_eucliden_acc_o = 0.0
avg_cosine_acc_o = 0.0
avg_MRR_o = 0.0
avg_hit_o = 0

for fold_num in range(0,1):
	sess_attention.run(init)
	sess_original.run(init_o)
	sess_calculate.run(init_cal)

	print("\n\n******************************************************")
	print(str(fold_num+1)+' Time for Loading Data and Vector...')

# ***************
# Loading DataSet
# ***************
	test_data = []
	MRR_test_data = []
	train_data = []
	loadcount = 0
	with open('./Data/'+filenameData+str(fold_num),'r') as fp:
		for line in fp:
			if (loadcount % 200) == 0:
				test_data.append(line)
				MRR_test_data.append(line)
			loadcount = loadcount + 1
	loadcount = 0
	for i in range(10):
		if i != fold_num:
			with open('./Data/'+filenameData+str(i),'r') as fp:
				for line in fp:
					if (loadcount % 2) == 0:
						train_data.append(line)
					loadcount = loadcount + 1

	print('Loading Data ...')
	test, t_target = function_files.loadData(train_data,wordvec,0,batch_size)
	print('Loading Data Completed')

	# Number of tuple of one batch
	num_lines = len(train_data)
	print('Train / Test: ' + str(num_lines) + '/' + str( len(test_data) ) )


# ********
# Training
# ********
	if __name__ == '__main__':
		print('------Training With Attention------')
		count = 0
		itr_count = 0
		prev_error = 0.0
		error_now = 0.0
		for epoch in range(int(training_iters)):
			for _ in range(10):
				try:
					train, label = function_files.loadData(train_data,wordvec,(batch_size*count)%num_lines,batch_size)
					count = count + 1

					sess_attention.run([optimizer],{data: train, target: label, keep_prob_ph: KEEP_PROB})


				except Exception as e:
					print("Error : {0}".format(str(e.args[0])).encode("utf-8"))
					itr_count = 0
			
			#print epoch,count
			try:
				error = sess_attention.run(loss, {data: test, target: t_target, keep_prob_ph: KEEP_PROB})
			except Exception as e:
				print("Error : {0}".format(str(e.args[0])).encode("utf-8"))
			diff = abs(error-prev_error)

			prev_error = error
			
			if ( (epoch+1) % 10) == 0:
				print('Epoch {:2d} error {:3.10f} '.format(epoch + 1, error))

			if diff < threshold_loss:
				itr_count = itr_count + 1
			else:
				itr_count = 0

			if itr_count > itr_notChange:
				print("Exit By converted")
				break
			
			error_now = error

		saveModel(sess_attention,savePath)
		avg_last_epoch_error = avg_last_epoch_error + error_now

		print('------Training Without Attention------')
		count = 0
		itr_count = 0
		prev_error = 0.0
		error_now = 0.0
		for epoch in range(int(training_iters)):
			for _ in range(10):
				try:
					train, label = function_files.loadData(train_data,wordvec,(batch_size*count)%num_lines,batch_size)
					count = count + 1

					sess_original.run([optimizer_o],{data_o: train, target_o: label})

				except Exception as e:
					print("Error : {0}".format(str(e.args[0])).encode("utf-8"))
					itr_count = 0
			
			#print epoch,count
			try:
				error = sess_original.run(loss_o, {data_o: test, target_o: t_target})
			except Exception as e:
				print("Error : {0}".format(str(e.args[0])).encode("utf-8"))
			diff = abs(error-prev_error)

			prev_error = error

			if ( (epoch+1) % 10) == 0:
				print('Epoch {:2d} error {:3.10f} '.format(epoch + 1, error))
			
			if diff < threshold_loss:
				itr_count = itr_count + 1
			else:
				itr_count = 0

			if itr_count > itr_notChange:
				print("Exit By converted")
				break

			error_now = error
		
		saveModel_o(sess_original,savePath_o)
		avg_last_epoch_error_o = avg_last_epoch_error_o + error_now


# *******
# Testing
# *******
	print('\n------Calculating Soecific Words------')
	test_data=['DBPEDIA_ID/hot_dog	hot dog','DBPEDIA_ID/red_tape	red tape','DBPEDIA_ID/white_elephant	white elephant','DBPEDIA_ID/real_estate	real estate',
	'DBPEDIA_ID/san_francisco	san francisco','DBPEDIA_ID/new_york_times	new york times','DBPEDIA_ID/donald_trump	donald trump','DBPEDIA_ID/harry_potter	harry potter',
	'DBPEDIA_ID/wet_blanket	wet blanket','DBPEDIA_ID/dutch_uncle	dutch uncle','DBPEDIA_ID/double_espresso	double espresso','DBPEDIA_ID/green_bushy_tree	green bushy tree',
	'DBPEDIA_ID/video_games	video games','DBPEDIA_ID/golden_gate_bridge	golden gate bridge','DBPEDIA_ID/alcatraz_island	alcatraz island','DBPEDIA_ID/tokyo_station	tokyo station',
	'DBPEDIA_ID/legal_document	legal document','DBPEDIA_ID/computer_science	computer science','DBPEDIA_ID/film_director	film director','DBPEDIA_ID/green_bird	green bird',
	'DBPEDIA_ID/tennis_shoe	tennis shoe'
	]
	dicts={}
	if __name__ == '__main__':	   
		for t in test_data:
			if not function_files.checkExist(t,wordvec):
				continue			
			test, avg_vec, ref_vec = function_files.loadDataTest(t,wordvec)
			rnn_vec = sess_attention.run(prediction, {data :test, keep_prob_ph: 1.0})
			rnn_vec_o = sess_original.run(prediction_o, {data_o :test})
			key = str(t).split('DBPEDIA_ID/')[1].split('\t')[0]

			rn_o = rnn_vec_o[0].tolist()
			rn = rnn_vec[0].tolist()
			av = avg_vec.tolist()
			dicts.update({key:	{  'avg':av,'rnn':rn_o,'rnn_att':rn  } 	})

		with open('specific_word_vec.json', 'w') as outfile:  
			json.dump(dicts,outfile)


sess_attention.close()
sess_original.close()