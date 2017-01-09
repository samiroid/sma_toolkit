from collections import Counter, defaultdict
import csv
from ipdb import set_trace
import numpy as np
import os
  
def colstr(string, color, best):
    # set_trace()
    if color is None:
        cstring = string
    elif color == 'red':
        cstring = "\033[31m" + string  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + string  + "\033[0m"

    if best: 
        cstring += " ** "
    else:
        cstring += "    "

    return cstring    
    
def word_2_idx(msgs, zero_for_padd=False, max_words=None):
    """
        Compute a dictionary index mapping words into indices
    """ 
    words = [w for m in msgs for w in m.split()]
    if max_words is not None:                
        top_words = sorted(Counter(words).items(), key=lambda x:x[1],reverse=True)[:max_words]                    
        words = [w[0] for w in top_words]
    #prepend the padding token
    if zero_for_padd: words = ['_pad_'] + list(words)    
    return {w:i for i,w in enumerate(set(words))}

def kfolds(n_folds,n_elements,val_set=False,shuffle=False,random_seed=1234):        
    if val_set:
        assert n_folds>2
    
    X = np.arange(n_elements)
    if shuffle: 
        rng=np.random.RandomState(random_seed)      
        rng.shuffle(X)    
    X = X.tolist()
    slice_size = n_elements/n_folds
    slices =  [X[j*slice_size:(j+1)*slice_size] for j in xrange(n_folds)]
    #append the remaining elements to the last slice
    slices[-1] += X[n_folds*slice_size:]
    kf = []
    for i in xrange(len(slices)):
        train = slices[:]
        # from pdb import set_trace; set_trace()
        # print i
        test = train.pop(i)
        if val_set:
            try:
                val = train.pop(i)
            except IndexError:
                val = train.pop(-1)                
            #flatten the list of lists
            train = [item for sublist in train for item in sublist]
            kf.append([train,test,val])
        else:
            train = [item for sublist in train for item in sublist]
            kf.append([train,test])
    return kf

def build_folds(msg_ids, n_folds, folder_path):
    """
        Compute partitions for crossfold validation
        and write the splits into CSV files 
        Each CSV file will contain the msg ids

    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    kf = kfolds(n_folds, len(msg_ids),val_set=True,shuffle=True)
    for i, fold in enumerate(kf):
        fold_data = {"train": [ int(msg_ids[x]) for x in fold[0] ], 
                     "test" : [ int(msg_ids[x]) for x in fold[1] ],
                     "val"  : [ int(msg_ids[x]) for x in fold[2] ] }
        with open(folder_path + '/fold_%d.csv' % i, 'wb') as f: 
            w = csv.DictWriter(f, fold_data.keys())
            w.writeheader()
            w.writerow(fold_data)

def shuffle_split(data, split_perc = 0.8, random_seed=1234):
    """
        Split the data into train and test, keeping the class proportions

        data: list of (x,y) tuples
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles

        returns: balanced training and test sets
    """
    rng=np.random.RandomState(random_seed)          
    z = defaultdict(list)
    #shuffle data
    rng.shuffle(data)
    #group examples by class label    
    z = defaultdict(list)
    for x,y in data: z[y].append(x)    
    train = []    
    test  = []
    for label in z.keys():
        #examples of each label 
        x_label  = z[label]            
        train += zip(x_label[:int(len(x_label)*split_perc)],
                    [label] * int(len(x_label)*split_perc))         
        test  += zip(x_label[ int(len(x_label)*split_perc):],
                    [label] * int(len(x_label)*(1-split_perc)))
    #reshuffle
    rng.shuffle(train)
    rng.shuffle(test)    

    return train, test

def stratified_sampling(data, n, random_seed=1234):
    """
        Get a sample of the data, keeping the class proportions

        data: list of (x,y) tuples
        n: number of samples
        random_seed: ensure repeatable shuffles

        returns: balanced sample
    """
    rng=np.random.RandomState(random_seed)          
    z = defaultdict(list)
    #shuffle data
    rng.shuffle(data)
    #group examples by class    
    z = defaultdict(list)    
    for x,y in data: z[y].append(x)    
    #compute class distribution
    class_dist = {}
    for cl, samples in z.items():
        class_dist[cl] = int((len(samples)*1./len(data)) * n)
    train = []    
    
    for label in z.keys():
        #examples of each label 
        x_label  = z[label]            
        train += zip(x_label[:class_dist[label]],
                    [label] * class_dist[label])             
    #reshuffle
    rng.shuffle(train)
    return train

