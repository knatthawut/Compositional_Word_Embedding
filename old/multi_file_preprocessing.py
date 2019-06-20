import subprocess
import glob
import sys

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
    filenames = glob.glob('./{}_tmp*'.format(input_file))
    for filename in filenames:
        cmd = 'python preprocessing.py {} {}'.format(filename,filename+'.out')
        return_value = subprocess.call(cmd,shell=True)

    #cat file
    filenames = glob.glob('./{}_tmp*.out'.format(input_file))
    cmd = 'cat'
    for filename in filenames:
        cmd = cmd + ' ' + filename
    cmd = cmd + ' > {}'.format(output_file)
    return_value = subprocess.call(cmd,shell=True)

if __name__ == '__main__':
    main()
