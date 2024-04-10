import re
import os
from multiprocessing import Process
import subprocess
import glob
import sys
# import preprocessing
import preprocessing_encow
def main():
    if len(sys.argv) < 3:
        print('Usages input_file output_file')
        sys.exit()
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])

    #splits file into files each file = 1M lines
    cmd = 'split -d -l 1000000 {} {}_tmp'.format(input_file,input_file)
    return_value = subprocess.call(cmd,shell=True)

    # run all files
    processes = []
    filenames = glob.glob('./{}_tmp*'.format(input_file))
    for filename in filenames:
        out_filename = filename + '.out'
        process = Process(target=preprocessing_encow.process(filename,out_filename),args=(filename,out_filename))
        processes.append(process)

        process.start()

    #cat file
    # cmd = 'cat {}_tmp*.out > {}'.format(input_file,output_file)
    # return_value = subprocess.call(cmd,shell=True)

if __name__ == '__main__':
    main()
