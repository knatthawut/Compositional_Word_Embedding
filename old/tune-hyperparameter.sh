#!/bin/sh
set -x
for bs in 16 32
do
	echo "================================ Tuning Hyperparameter ========================"
	python Tratz_classification.py --baseline AVG --type_of_Word2Vec_model CBOW --vector_file_path ../model/encow14_wiki_CBOW_size200_window5_min50 --num_of_epoch 10000  --num_of_epoch_composition 1 --batch_size $bs --batch_size_composition 16384 --activation_func tanh --lr 1e-2 --tensorboard_path logs/AVG_new_lr_1e-2_10000epoch_comp1_bs$bs > ./results/AVG_new_lr_1e-2_20000epoch_comp1_bs$bs.out
done
