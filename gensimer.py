import argparse
import cPickle
from collections import Counter
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
import os
import time
from __init__ import word_2_idx

class Word2VecReader(object):
	def __init__(self, datasets, max_sent=None):
		self.datasets = datasets
		self.max_sent = max_sent if max_sent else float('inf')
		if self.max_sent < float('inf'):
			print "[max_sentences: %d]" % self.max_sent
	def __iter__(self):		
		for dataset in self.datasets:
			lines=0	
			with open(dataset) as fid:
				for l in fid:		
					lines+=1
					if lines>self.max_sent: break
					yield l.decode("utf-8").split()

class Doc2VecReader(object):
	"""
		IMPORTANT: this reader assumes that each line in the file has the structure <paragraph_id>\t<sentence>
		All sentences with the same <paragrah_id> will be considered as one paragraph
	"""
	def __init__(self, datasets, max_sent=None):
		self.datasets = datasets
		self.max_sent = max_sent if max_sent else float('inf')
		if self.max_sent < float('inf'):
			print "[max_sentences: %d]" % self.max_sent
		self.doc2idx = {}

	def load_model(self, model_path, doc2idx_path):
		self.d2v = Doc2Vec.load(model_path)
		with open(doc2idx_path) as fid:
			self.doc2idx = cPickle.load(fid)

	def save_idx(self, path):
		with open(path,"w") as fod:
			cPickle.dump(self.doc2idx, fod, -1)

	def vector(self, doc_id):
		assert self.d2v is not None, "Did your forget to load_model() ?"		
		idx = self.doc2idx[doc_id]
		return self.d2v.docvecs[idx]

	def infer_vector(self, doc):
		assert self.d2v is not None, "Did your forget to load_model() ?"
		return self.d2v.infer_vector(doc)

	def __iter__(self):		
		for dataset in self.datasets:
			print dataset			
			with open(dataset) as fid:				
				for i, l in enumerate(fid):						
					if i>self.max_sent: break					
					splt = l.decode("utf-8").split()
					if len(splt)<2: continue
					doc_id = splt[0]
					try:
						idx = self.doc2idx[doc_id]
					except KeyError:
						idx=len(self.doc2idx)
						self.doc2idx[doc_id]=idx
					txt = splt[1:]
					yield LabeledSentence(words=txt,tags=[idx])


class LDAReader(object):
	def __init__(self, datasets, max_sent=None):
		"""
			datasets: datasets
			max_sent: maximum number of sentences to be read in each dataset			
		"""
		self.datasets = datasets
		self.max_sent = max_sent if max_sent else float('inf') 
		if self.max_sent < float('inf'):
			print "[max_sentences: %d]" % self.max_sent
		self.wrd2idx = None
		self.idx2wrd = None		
		self.model   = None

	def load_vocabulary(self, wrd2idx):
		self.wrd2idx = wrd2idx
		self.idx2wrd = {i:w for w,i in self.wrd2idx.items()}
		

	def save_vocabulary(self, path):
		with open(path,"w") as fod:
			cPickle.dump(self.wrd2idx, fod) 
			print "[wrd2idx saved @ %s]" % path

	def load_model(self, model_path, wrd2idx_path):
		self.model = LdaMulticore.load(model_path)
		with open(wrd2idx_path) as fid:
			wrd2idx = cPickle.load(fid)		
		self.load_vocabulary(wrd2idx)

	def get_topics(self, doc, binary=False):
		assert self.model is not None, "Model not found! Please did you forget to load_model() ?"
		feats = np.zeros(self.model.num_topics)
		topics = self.model.get_document_topics(self.features(doc))
		for t in topics:
			feats[t[0]] = t[1] if binary else 0
		return feats

	def compute_vocabulary(self):
		ct = Counter()
		for dataset in self.datasets:
			print dataset
			with open(dataset) as fid:
				for i,l in enumerate(fid):
					if i > self.max_sent: break
					ct.update(l.decode("utf-8").split())		
		self.wrd2idx = {w:i for i,w in enumerate(ct.keys())}
		self.idx2wrd = {i:w for w,i in self.wrd2idx.items()}
	
	def features(self, doc):
		ct = Counter(doc.split())
		return [(self.wrd2idx[w],c) for w,c in ct.items() if w in self.wrd2idx]

	def __iter__(self):		
		assert self.wrd2idx is not None and self.idx2wrd is not None, "Vocabulary not found! Did you forget to call compute_vocabulary() ?"
		for dataset in self.datasets:
			print dataset			
			with open(dataset) as fid:
				for i,l in enumerate(fid):					
					if i>self.max_sent: break
					yield self.features(l.decode("utf-8"))

def train_lda(args):
	print "[LDA > n_topics: %d ]" % args.dim	
	lda_reader = LDAReader(args.ds, max_sent=args.max_sent)		
	lda_reader.compute_vocabulary()	
	lda_model = LdaMulticore(lda_reader, id2word=lda_reader.idx2wrd,
									   num_topics=args.dim, 
									   workers=args.workers)
	lda_model.save(args.out)	
	idx_path =  os.path.splitext(args.out)[0]+"_idx.pkl"
	lda_reader.save_vocabulary(idx_path)
	

def train_skipgram(args):
	t0 = time.time()
	if args.negative_samples > 0:		
		print "[SKip-Gram > negative_samples: %d | min_count: %d | dim: %d | epochs: %d]" % (args.negative_samples, args.min_count, args.dim, args.epochs)
		w2v = Word2Vec(sentences=Word2VecReader(args.ds,args.max_sent), size=args.dim, 
			           workers=args.workers, min_count=args.min_count, sg=1, hs=0, 
			           negative=args.negative_samples, iter=args.epochs)		
	else:		
		print "[SKip-Gram (Hierachical Softmax) > min_count: %d | dim: %d | epochs: %d]" % (args.min_count, args.dim, args.epochs)
		w2v = Word2Vec(sentences=Word2VecReader(args.ds,args.max_sent), size=args.dim, 
			           workers=args.workers, min_count=args.min_count, sg=0, 
			           hs=1,iter=args.epochs)		
	w2v.train(Word2VecReader(args.ds,args.max_sent))
	path_out = os.path.splitext(args.out)[0]
	w2v.save(path_out+".pkl")
	w2v.save_word2vec_format(path_out+".txt")
	tend = time.time() - t0
	mins = np.floor(tend*1./60)
	secs = tend - mins*60
	print "[runtime: %d.%d mins]" % (mins,secs)
	print "Done"	

def train_doc2vec(args):
	d2v_reader = Doc2VecReader(args.ds,args.max_sent)
	if args.negative_samples > 0:		
		print "[Doc2Vec > negative_samples: %d | min_count: %d | dim: %d | epochs: %d]" % (args.negative_samples, args.min_count, args.dim, args.epochs)		
		d2v = Doc2Vec(documents=d2v_reader, size=args.dim, 
			           workers=args.workers, min_count=args.min_count, hs=0, 
			           negative=args.negative_samples, iter=args.epochs)		
	else:		
		print "[Doc2Vec (Hierachical Softmax) > min_count: %d | dim: %d | epochs: %d]" % (args.min_count, args.dim, args.epochs)
		d2v = Doc2Vec(documents=d2v_reader, size=args.dim, 
			           workers=args.workers, min_count=args.min_count, 
			           hs=1,iter=args.epochs)		
	d2v_reader = Doc2VecReader(args.ds,args.max_sent)
	d2v.train(d2v_reader)		
	d2v.save(args.out)	
	idx_path =  os.path.splitext(args.out)[0]+"_idx.pkl"
	d2v_reader.save_idx(idx_path)
	print "Done"	

def get_parser():
	parser = argparse.ArgumentParser(description="Induce Text Representations with Gensim")
	parser.add_argument('-ds',  type=str, required=True, nargs='+', help='datasets')        
	parser.add_argument('-out', type=str, required=True, help='path to store the embeddings')
	parser.add_argument('-dim', type=int, required=True, help='size of embeddings or number of topics')
	parser.add_argument('-model',    choices=['w2v','doc2vec','lda'], required=True, help='model')
	parser.add_argument('-epochs',   type=int, default=5, help='number of epochs')
	parser.add_argument('-workers',  type=int, default=4, help='number of workers')
	parser.add_argument('-max_sent', type=int, help='set max number of sentences to be read (per file)')
	parser.add_argument('-min_count',type=int, default=10, help='words ocurring less than ''min_count'' times are discarded')
	parser.add_argument('-negative_samples', type=int, default=10, help='number of negative samples for Skip-Gram training. If set to 0 then Hierarchical Softmax will be used')
	return parser


if __name__ == "__main__":	
	cmdline_parser = get_parser()
	args = cmdline_parser.parse_args() 			
	print "** Induce Text Representations with Gensim **"
	print "[input > %s | max_sent: %s | workers: %d | output@%s]\n" % (repr(args.ds), repr(args.max_sent), args.workers,  args.out)	

	#create output folder if it does not exist
	out_folder = os.path.dirname(args.out)
	if not os.path.exists(out_folder):		
		os.makedirs(out_folder)
		print "[created output folder: %s]" % out_folder		

	if args.model =="lda":
		train_lda(args)
	elif args.model == "doc2vec":
		train_doc2vec(args)
	elif args.model == "w2v":
		train_skipgram(args)
	else:		
		raise NotImplementedError, "unknown model: %s" % args.model
