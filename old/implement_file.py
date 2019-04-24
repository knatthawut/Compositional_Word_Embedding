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

# hyperparmeters
learning_rate = 0.0003
training_iters = 5000
batch_size = 50000

KEEP_PROB = 0.58

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
filenameData = 'uni_pair_combine_less10_'

vectorFile = './'+type_model+'/wiki-db_more50_'+str(n_input)# sys.argv[1]
save_load_Path = "./model/"+str(n_input)+"/attention_machenism"

loadFlag = False

finalResultFile = str(n_input)+'_'+type_model+'.result'


# ****************
# load Word Vector
# ****************

wordvec = function_files.wordvec
wordvec.init_sims(replace=True)
print('---> Finish Loading <---\n')


def lengths(data):
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
	


print('\n---> Setting Tensorflow <---')
# Set placeholder
data = tf.placeholder(tf.float32, [None, max_length, n_input])
target = tf.placeholder(tf.float32, [None, n_classes])
keep_prob_ph = tf.placeholder(tf.float32)
	

num_layers = 2


# Recurrent network.
output, last = rnn.dynamic_rnn(
	rnn_cell.GRUCell(num_hidden),
	data,
	dtype=tf.float32,
	sequence_length=lengths(data),
	time_major=False
)


# ***************
# Attention Layer
# ***************
attention_size=180

output_shape = output.shape
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


loss = tf.reduce_mean(  tf.square(target - prediction)  )
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
   



avg_last_epoch_error = 0
avg_eucliden_acc = 0
avg_cosine_acc = 0
avg_MRR = 0
for fold_num in range(0,10):

	if loadFlag:
		saver.restore(sess,save_load_Path)
	else:
		sess.run(tf.global_variables_initializer())

	test_data = []
	MRR_test_data = []
	train_data = []


# ------------------------------------------------------------------------
# loading train_data & test_data
# ------------------------------------------------------------------------
	# text Data
	print("\n\n**********************************")
	print(str(fold_num+1)+' Time for Loading Data and Vector...')
	loadcount = 0
	with open('./Data/'+filenameData+str(fold_num),'r', encoding = 'utf8') as fp:
		for line in fp:
			if (loadcount % 200) == 0:
				test_data.append(line)
				MRR_test_data.append(line)
			loadcount = loadcount + 1
	loadcount = 0
	for i in range(10):
		if i != fold_num:
			with open('./Data/'+filenameData+str(i),'r', encoding = 'utf8') as fp:
				for line in fp:
					if (loadcount % 5) == 0:
						train_data.append(line)
					loadcount = loadcount + 1
	
	test, t_target = loadData(train_data,wordvec,0,batch_size)
	print('Loading Data Completed')

	# Number of tuple of one batch
	num_lines = len(train_data)
	count = 0
	print("N Sample :", str(num_lines))


# ------------------------------
# training
# ------------------------------
	if __name__ == '__main__':
		# Training
		itr_count = 0
		prev_error = 0.0
		for epoch in range(int(training_iters/10)):
			for _ in range(10):
				try:
					train, label = loadData(train_data,wordvec,(batch_size*count)%num_lines,batch_size)
					count = count + 1
					sess.run([optimizer],{data: train, target: label, keep_prob_ph: KEEP_PROB})
				except Exception as e:
					print("Error : {0}".format(str(e.args[0])).encode("utf-8"))
					itr_count = 0
			
			#print epoch,count
			error = sess.run(loss, {data: test, target: t_target, keep_prob_ph: KEEP_PROB})
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
		
		save_path = saver.save(sess,save_load_Path)
		print('\nSaving Model at:'+save_path)
		

# ------------------------------
# testing
# ------------------------------
	  
	if __name__ == '__main__':	   
	# Testing
		eucliden_acc = 0.0
		cosine_acc = 0.0
		counter = 0.0
		for t in test_data:
			if not function_files.checkExist(t,wordvec):
				continue
			counter = counter + 1
			test, avg_vec, ref_vec = loadDataTest(t,wordvec)
			rnn_vec = sess.run(prediction, {data :test, keep_prob_ph: 1.0})
			if np.linalg.norm(avg_vec - ref_vec) > np.linalg.norm(rnn_vec - ref_vec):
				eucliden_acc = eucliden_acc + 1.0
			if (1-scipy.spatial.distance.cosine(rnn_vec, ref_vec)) > (1-scipy.spatial.distance.cosine(avg_vec, ref_vec)):
				cosine_acc = cosine_acc + 1.0
		print(eucliden_acc/counter,cosine_acc/counter)
	#	 with open(finalResultFile,'a') as fr:
	#		 fr.write(str(fold_num)+'\t'+str(eucliden_acc/counter)+'\t'+str(cosine_acc/counter)+'\t'+str(eucliden_acc)+'\t'+str(cosine_acc)+'\t'+str(counter)+'\n')

		avg_eucliden_acc = avg_eucliden_acc + (eucliden_acc/counter)
		avg_cosine_acc = avg_cosine_acc + (cosine_acc/counter)	   


# ------------------------------
# MRR
# ------------------------------
	print('\n---> Calculating MRR <---')
	p_value = []
	if __name__ == '__main__': 
		summary = 0.0
		count = 0
		print('length:'+str(len(MRR_test_data)))
		leng = str(len(MRR_test_data))
		for t in MRR_test_data:
			if not function_files.checkExist(t,wordvec):
				continue
			rk, avg_vec, ref_vec = loadDataTest(t,wordvec)
			rnn_vec = sess.run(prediction, feed_dict={data :rk, keep_prob_ph: 1.0})
			
			rank = function_files.ranking(ref_vec,rnn_vec,wordvec)
			
			p_value.append(str(rank))

			summary = summary + rank
			count = count + 1
			if (count % 100) == 0:
				print(str(count)+"/"+str(leng))
				pp.pprint('-MR Right Now:'+str(summary/float(count)))
		with open('MRR/p_value_'+str(n_input)+'_'+str(fold_num)+'_'+'last.json', 'w') as outfile:
			json.dump(p_value, outfile)

		avg_MRR = avg_MRR + (summary/float(count))
	
	print("**********************************")
		


print("avg_last_epoch_error:"+str(avg_last_epoch_error/10) )
print("avg_eucliden_acc:"+str(avg_eucliden_acc/10) )
print("avg_cosine_acc:"+str(avg_cosine_acc/10) )
print("avg_MRR:"+str(avg_MRR/10) )
