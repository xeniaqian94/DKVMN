import numpy as np
import sys

'''
Usage: python split_data_532.py ../../data/duolingo/token_unsplitted_train.csv
'''

def load_data(path):
    f_data = open(path , 'r')
    q_data = []
    qa_data = []
    dest_list = ["train1", "valid1", "test"]
    f_list=[]
    for name in dest_list:
        f_list+=[open(path.replace("train",name),"w")]

    for lineID, line in enumerate(f_data):

        line = line.strip( )
        # lineID starts from 0
        if lineID % 3 == 0:
            destination=np.random.choice(3, p=[0.5, 0.3, 0.2])

        f_list[destination].write(line+"\n")

if __name__ == '__main__':
    load_data(sys.argv[1])