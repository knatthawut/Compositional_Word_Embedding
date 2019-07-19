#!/bin/bash
echo -n "Input the path to the input_file: "
read Input_file
echo -n "Input the path to the output_file: "
read Out_file
Tmp_file='process.tmp'
python multi_file_processing.py $Input_file $Tmp_file
python generate_word_vector_train_data.py $Tmp_file $Out_file
echo "Done preprocessing!"
echo "Result file is saved at: "$Out_file