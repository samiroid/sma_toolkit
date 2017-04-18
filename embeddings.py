import codecs
import cPickle 
from __init__ import idx_2_wrd
import numpy as np
import theano 
import theano.tensor as T

class Emb_Mapper():

    def __init__(self, Emb_in, Emb_out,lrate=0.01):
        #initializations
        rng = np.random.RandomState(1234)        
        I = theano.shared(Emb_in.astype(theano.config.floatX))
        O = theano.shared(Emb_out.astype(theano.config.floatX))
        self.W = self.init_W((Emb_out.shape[1],Emb_in.shape[1]), rng)               
        #model
        # from ipdb import set_trace; set_trace()
        x = T.iscalar('x')
        x_in = I[x,:]
        x_out = O[x,:]
        hat_x_out = T.dot(self.W,x_in)
        diff = hat_x_out - x_out
        #cost
        J = (diff ** 2).sum()
     
        grad_W = T.grad(J,self.W) 
        updates = ((self.W, self.W - lrate * grad_W),)
        self.train = theano.function(inputs=[x],                                
                                      outputs=J,
                                      updates=updates)
    
    def init_W(self, size, rng):                
        W = np.asarray(rng.uniform(low=-1, high=1, size=size))
        return theano.shared(W.astype(theano.config.floatX), borrow=True)

    def save(self, path):
        with open(path,"wb") as fod:
            cPickle.dump(self.W.get_value(), fod, cPickle.HIGHEST_PROTOCOL)

# def embeddings_to_dict(path, max_words=None):
#     """
#         Read word embeddings into a dictionary
#     """
#     w2v = {}
#     data = None
#     with codecs.open(path,"r","utf-8") as fid:
#         fid.readline()        
#         data = fid.readlines()
#     #avoid extra comparisons if we want load all the words
#     if max_words is None:
#         for line in data.split("\n"):
#             entry = line.split()
#             if len(entry) > 2:
#                 w2v[entry[0]] = np.array(entry[1:]).astype('float32')
#     else:
#         for i, line in enumerate(data.split("\n")):
#             entry = line.split()
#             if len(entry) > 2:
#                 w2v[entry[0]] = np.array(entry[1:]).astype('float32')
#             if i >= max_words:break
#     return w2v   

def embeddings_to_dict(path, max_words=None):
    """
        Read word embeddings into a dictionary
    """
    w2v = {}    
    with codecs.open(path,"r","utf-8") as fid:
        #ignore first line
        fid.readline()                
        #avoid extra comparisons if we want load all the words
        if max_words is None:
            for line in fid:
                entry = line.split()
                if len(entry) > 2:
                    w2v[entry[0]] = np.array(entry[1:]).astype('float32')
        else:
            for i, line in enumerate(fid):
                entry = line.split()
                if len(entry) > 2:
                    w2v[entry[0]] = np.array(entry[1:]).astype('float32')
                if i >= max_words:break
    return w2v 

def read_embeddings(path, wrd2idx=None, max_words=None):

    w2v = embeddings_to_dict(path,max_words)        
    #if no word index is specified read all the embedding vocabulary
    if wrd2idx is None:
        wrd2idx = {w:i for i,w in enumerate(w2v.keys())}                
    #build embedding matrix
    emb_size = w2v.values()[0].shape[0]    
    E = np.zeros((emb_size, len(wrd2idx)))    
    for w,i in wrd2idx.items():
        if w in w2v: E[:,i] = w2v[w]            
    # Number of out of embedding vocabulary embeddings
    n_OOEV = np.sum((E.sum(0) == 0).astype(int))
    perc = n_OOEV*100./len(wrd2idx)
    print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
           % (n_OOEV, len(wrd2idx), perc)) 

    ooev_idx = np.where(~E.any(axis=0))[0]
    idx2wrd = idx_2_wrd(wrd2idx)
    ooevs = [idx2wrd[idx] for idx in ooev_idx]
    
    return E, wrd2idx, ooevs

def filter_embeddings(path_in, path_out, wrd2idx):

    """
        Filter embeddings file to contain only the relevant set
        of words (so that it can be loaded faster)
    """

    w2v = embeddings_to_dict(path_in)  
    ooevs = wrd2idx.copy()
    with codecs.open(path_out,"w","utf-8") as fod:
        voc_size = len(wrd2idx)
        emb_size = w2v.values()[0].shape[0]   
        fod.write(str(voc_size)+"\t"+str(emb_size)+"\n")
        for word,emb in w2v.items():
            if word in wrd2idx:
                del ooevs[word]                
                fod.write("%s %s\n" % (word, " ".join(map(str, emb))))
        perc = len(ooevs)*100./len(wrd2idx)
        print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
           % (len(ooevs), len(wrd2idx), perc)) 
        #ooev words
        return ooevs

def save_txt(path, E, wrd2idx):
    with open(path,"w") as fod:
        fod.write("%d %d\n" % (E.shape[1],E.shape[0]))  
        for word, idx in wrd2idx.items():      
            emb = E[:,idx]
            fod.write("%s %s\n" % (word, " ".join(map(str, emb))))

# def filter_embeddings(path_in, path_out, wrd2idx):

#     """
#         Filter embeddings file to contain only the relevant set
#         of words (so that it can be loaded faster)
#     """
#     emb_values = []
#     ooevs = wrd2idx.copy()
#     with codecs.open(path_out,"w","utf-8") as fod:
#         with codecs.open(path_in,"r","utf-8") as fid:
#             voc_size = len(wrd2idx)
#             _, emb_size = fid.readline().split()        
#             # emb_values = np.zeros(int(emb_size))
#             fod.write(str(voc_size)+"\t"+str(emb_size)+"\n")
#             for line in fid.readlines():
#                 items = line.split()
#                 wrd   = items[0]
#                 if wrd in wrd2idx:
#                     del ooevs[wrd]
#                     emb_values.append(np.array(items[1:]).astype(float))
#                     fod.write(line)
#         perc = len(ooevs)*100./len(wrd2idx)
#         print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
#            % (len(ooevs), len(wrd2idx), perc)) 
#         #ooev words
#         return ooevs

# def get_embeddings(path, wrd2idx, max_words):

#     """
#         Recover an embedding matrix consisting of the relevant
#         vectors for the given set of words
#     """
#     with codecs.open(path,"r","utf-8") as fid:
#         voc_size = len(wrd2idx)        
#         _, emb_size = fid.readline().split()        
#         E = np.zeros((int(emb_size), voc_size))
#         for line in fid.readlines():
#             items = line.split()
#             if len(items) > 2: 
#                 wrd   = items[0]
#                 if wrd in wrd2idx:
#                     E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)
#     # Number of out of embedding vocabulary embeddings
#     n_OOEV = np.sum((E.sum(0) == 0).astype(int))
#     perc = n_OOEV*100./len(wrd2idx)
#     print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
#            % (n_OOEV, len(wrd2idx), perc)) 

#     ooev_idx = np.where(~E.any(axis=0))[0]
#     idx2wrd = idx_2_wrd(wrd2idx)
#     ooevs = [idx2wrd[idx] for idx in ooev_idx]

#     return E, ooevs

# def get_embeddings(path, wrd2idx, max_words):

#     """
#         Recover an embedding matrix consisting of the relevant
#         vectors for the given set of words
#     """
#     with codecs.open(path,"r","utf-8") as fid:
#         voc_size = len(wrd2idx)        
#         _, emb_size = fid.readline().split()        
#         E = np.zeros((int(emb_size), voc_size))
#         for line in fid.readlines():
#             items = line.split()
#             if len(items) > 2: 
#                 wrd   = items[0]
#                 if wrd in wrd2idx:
#                     E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)
#     # Number of out of embedding vocabulary embeddings
#     n_OOEV = np.sum((E.sum(0) == 0).astype(int))
#     perc = n_OOEV*100./len(wrd2idx)
#     print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
#            % (n_OOEV, len(wrd2idx), perc)) 

#     ooev_idx = np.where(~E.any(axis=0))[0]
#     idx2wrd = idx_2_wrd(wrd2idx)
#     ooevs = [idx2wrd[idx] for idx in ooev_idx]

#     return E, ooevs




