
import sys
import subprocess
import os
def main(compare_baseline):
    baselines = ['AVG','SimpleRNN','BiRNN','BiRNN_withoutDense','GRU','BiGRU','LSTM','BiLSTM','GRU_Attention','GRU_Attention_Multi','BiGRU_Attention','LSTM_Attention','BiLSTM_Attention','Conv1D']
    for baseline in baselines:
        if baseline == compare_baseline:
            continue
        command = 'CUDA_VISIBLE_DEVICES=1 python run_exp1.py --main_baseline {} --compare_baseline {}'.format(baseline,compare_baseline)
        os.system(command)


if __name__ == '__main__':
    if (len(sys.argv)<2):
        print('Usages comparison_baseline')
        sys.exit()
    compare_baseline = str(sys.argv[1])
    main(compare_baseline)
