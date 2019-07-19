
import sys
import subprocess
import os
def main(compare_baseline,gpu_id):
    baselines = ['AVG','SimpleRNN','BiRNN','BiRNN_withoutDense','GRU','BiGRU','LSTM','BiLSTM','GRU_Attention','GRU_Attention_Multi','BiGRU_Attention','LSTM_Attention','BiLSTM_Attention','Conv1D']
    for baseline in baselines:
        if baseline == compare_baseline:
            continue
        command = 'CUDA_VISIBLE_DEVICES={} python run_exp1.py --main_baseline {} --compare_baseline {}'.format(str(gpu_id),baseline,compare_baseline)
        os.system(command)


if __name__ == '__main__':
    if (len(sys.argv)!=3):
        print('Usages comparison_baseline GPU_ID')
        sys.exit()
    compare_baseline = str(sys.argv[1])
    gpu_id = int(sys.argv[2])
    main(compare_baseline,gpu_id)
