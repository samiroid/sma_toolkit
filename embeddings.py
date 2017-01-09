import numpy as np
import theano 
import cPickle 
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

def read_embeddings(path):

    w2v = embeddings_to_dict(path)    
    emb_size = w2v.values()[0].shape[0]
    n_items  = len(w2v) 
    E = np.zeros((emb_size, n_items))
    wrd2idx = {w:i for i,w in enumerate(w2v.keys())}
    for w,i in wrd2idx.items():
        E[:,i] = w2v[w]
    
    return E, wrd2idx

def get_embeddings(path, wrd2idx):

    """
        Recover an embedding matrix consisting of the relevant
        vectors for the given set of words
    """
    with open(path) as fid:
        voc_size = len(wrd2idx)        
        _, emb_size = fid.readline().split()        
        E = np.zeros((int(emb_size), voc_size))
        for line in fid.readlines():
            items = line.split()
            wrd   = items[0]
            if wrd in wrd2idx:
                E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)
    # Number of out of embedding vocabulary embeddings
    n_OOEV = np.sum((E.sum(0) == 0).astype(int))
    perc = n_OOEV*100./len(wrd2idx)
    print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
           % (n_OOEV, len(wrd2idx), perc)) 
    return E

def save_embeddings_txt(path_in, path_out, wrd2idx, init_ooe=False):

    """
        Filter embeddings file to contain only the relevant set
        of words (so that it can be loaded faster)

        init_ooe == True, then initialize out-of-embeddings with 
        samples from a multivariate gaussian with mean and covariance
        estimated from the embeddings that were found
    """
    emb_values = []
    all_words = wrd2idx.copy()
    with open(path_out,"w") as fod:
        with open(path_in,"r") as fid:
            voc_size = len(wrd2idx)
            _, emb_size = fid.readline().split()        
            # emb_values = np.zeros(int(emb_size))
            fod.write(str(voc_size)+"\t"+str(emb_size)+"\n")
            for line in fid.readlines():
                items = line.split()
                wrd   = items[0]
                if wrd in wrd2idx:
                    del all_words[wrd]
                    emb_values.append(np.array(items[1:]).astype(float))
                    fod.write(line)
        
        perc = len(all_words)*100./len(wrd2idx)
        print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" 
           % (len(all_words), len(wrd2idx), perc)) 

        if init_ooe:
            print "Initialize OOE words randomly"
            all_embs_vals = np.matrix(emb_values)  
            mu  = np.mean(all_embs_vals,axis=0)
            mu  = np.squeeze(np.asarray(mu))
            cov = np.cov(all_embs_vals,rowvar=0)
            sampled_embs = np.random.multivariate_normal(mu, cov,size=len(all_words))
            
            for i, wrd in enumerate(all_words):
                emb = sampled_embs[i,:]
                try:
                    fod.write("%s %s\n" % ( wrd.encode("utf-8"), " ".join(map(str, emb))) )   
                except UnicodeDecodeError:                   
                    try:
                        fod.write("%s %s\n" % ( wrd.decode("utf-8").encode("utf-8"), " ".join(map(str, emb))) ) 
                    except UnicodeDecodeError:                    
                        print "ERROR: ",  wrd

            
def embeddings_to_dict(path):
    """
        Read word embeddings into a dictionary
    """
    w2v = {}
    with open(path,"r") as fid:
        fid.readline()        
        for line in fid:
            entry = line.split()
            w2v[entry[0]] = np.array(entry[1:]).astype('float32')
    return w2v   
