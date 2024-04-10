#!/bin/sh
set -x
for bs in 4096 8192 16384
do
	echo "================================ Tuning Hyperparameter ========================"
	python Tratz_classification.py --baseline SimpleRNN --type_of_Word2Vec_model CBOW --vector_file_path ../model/encow14_wiki_CBOW_size200_window5_min50 --num_of_epoch 10000  --num_of_epoch_composition 1000 --batch_size 16 --batch_size_composition $bs --activation_func tanh --lr 1e-2 --tensorboard_path logs/SimpleRNN_new_lr_1e-2_10000epoch_comp1000_bs16_bscomp$bs > ./results/SimpleRNN_new_lr_1e-2_10000epoch_comp1000_bs16_bscomp$bs.out
done
