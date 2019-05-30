import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-ia', '--input_aus_filesdir', type=str, help='Dir with imgs aus files')
parser.add_argument('-op', '--output_path', type=str, help='Output path')
args = parser.parse_args()

def get_data(filepath):
    #data = dict()
    content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
    #data[os.path.basename(filepath[:-4])] = np.hstack([content[:,0:1].astype(np.int32), content[:, 5:22]])
    data = np.hstack([content[:,0:1].astype(np.int32), content[:, 5:22]])
    return data

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def main():
    filepaths = glob.glob(os.path.join(args.input_aus_filesdir, '*.csv'))
    filepaths.sort()
    print(filepaths)
    
    if not (os.path.exists(args.output_path) or os.path.isdir(args.output_path)):
        os.makedirs(args.output_path)
    for filepath in tqdm(filepaths):
        
        # create aus file
        data = get_data(filepath)
        name = os.path.basename(filepath[:-4])
        np.save(os.path.join(args.output_path, name+'.npy'), data)
        #save_dict(data, os.path.join(args.output_path, "aus_"+name))
        

if __name__ == '__main__':
    main()
