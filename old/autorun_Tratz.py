
import sys
import subprocess
import os
def main():
    baselines = ['AVG','Actual','Concatenate','SimpleRNN','BiRNN','BiRNN_withoutDense','GRU','BiGRU','LSTM','BiLSTM','GRU_Attention','GRU_Attention_Multi','BiGRU_Attention','LSTM_Attention','BiLSTM_Attention','Conv1D']
    for baseline in baselines:
        command = 'CUDA_VISIBLE_DEVICES=1 python Tratz_classification_withoutDB.py --baseline {} '.format(baseline)
        os.system(command)


if __name__ == '__main__':
    main()
