import argparse
import cPickle
from collections import Counter
from ipdb import set_trace
from __init__ import word_2_idx
from embeddings import save_embeddings
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
import os
import time

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
	lda_reader = LDAReader(args.input, max_sent=args.max_sent)		
	lda_reader.compute_vocabulary()	
	lda_model = LdaMulticore(lda_reader, id2word=lda_reader.idx2wrd,
									   num_topics=args.dim, 
									   workers=args.workers)
	lda_model.save(args.output)	
	idx_path =  os.path.splitext(args.output)[0]+"_idx.pkl"
	lda_reader.save_vocabulary(idx_path)
	

def train_skipgram(args):
	t0 = time.time()
	if args.negative_samples > 0:		
		print "[SKip-Gram > negative_samples: %d | min_count: %d | dim: %d | epochs: %d]" % (args.negative_samples, args.min_count, args.dim, args.epochs)
		w2v = Word2Vec(size=args.dim, 
			           workers=args.workers, min_count=args.min_count, sg=1, hs=0, 
			           negative=args.negative_samples, iter=args.epochs)		
	else:		
		print "[SKip-Gram (Hierachical Softmax) > min_count: %d | dim: %d | epochs: %d]" % (args.min_count, args.dim, args.epochs)
		w2v = Word2Vec(size=args.dim, 
			           workers=args.workers, min_count=args.min_count, sg=0, 
			           hs=1,iter=args.epochs)		
	w2v_reader = Word2VecReader(args.input,args.max_sent)
	w2v.build_vocab(w2v_reader)
	w2v.train(w2v_reader)
	path_out = os.path.splitext(args.output)[0]
	w2v.save(path_out+".pkl")
	w2v.wv.save_word2vec_format(path_out+".txt")
	tend = time.time() - t0
	mins = np.floor(tend*1./60)
	secs = tend - mins*60
	print "[runtime: %d.%d mins]" % (mins,secs)
	print "Done"	

def train_paragraph2vec(args):	
	#config model	
	dm=1 #this corresponds to the PV-DM model	
	dm_concat=0	
	if args.model == "pv-dbow":
		dm=0
	if args.model == "pv-dm-concat":		
		dm_concat=1			
	if args.negative_samples > 0:		
		print "[Doc2Vec > model: %s | word_vecs: %s | negative_samples: %d | min_count: %d | dim: %d | epochs: %d]" % (args.model, args.pretrained_vecs, args.negative_samples, args.min_count, args.dim, args.epochs)				
		hs=0 # no hierarchical softmax		
	else:		
		print "[Doc2Vec (Hierachical Softmax) > model: %s | word_vecs: %s | min_count: %d | dim: %d | epochs: %d]" % (args.model, args.pretrained_vecs, args.min_count, args.dim, args.epochs)		
		hs=1 # hierarchical softmax				
	d2v = Doc2Vec(size=args.dim, dm=dm, dm_concat=dm_concat,
				  hs=hs, negative=args.negative_samples, min_count=args.min_count, 
			      workers=args.workers, iter=args.epochs)
	#doc reader
	d2v_reader = Doc2VecReader(args.input,args.max_sent)	
	d2v.build_vocab(d2v_reader)			
	if args.pretrained_vecs:						
		print "[loading pre-trained vectors]"
		t0=time.time()
		d2v.intersect_word2vec_format(args.pretrained_vecs)
		tend = time.time() - t0
		mins = np.floor(tend*1./60)
		secs = tend - mins*60
		print "\r[loaded word vectors in: %d.%d mins]" % (mins,secs)				
	else:	
		print "[training word vectors]"	
		#also train word vectors
		d2v.dbow_words=1		
	
	d2v.train(d2v_reader)			
	d2v.save(args.output)		
	#build an embedding matrix with the paragraph vectors
	E = np.zeros((len(d2v.docvecs[0]),len(d2v.docvecs)))
	for idx, docvec in enumerate(d2v.docvecs):
		E[:,idx] = docvec
	save_embeddings(args.output+".txt", E, d2v_reader.doc2idx)
	d2v.wv.save_word2vec_format(args.output+"_words.txt")
	d2v.delete_temporary_training_data()
	print "Done"	

def get_parser():
	parser = argparse.ArgumentParser(description="Induce Text Representations with Gensim")
	parser.add_argument('-input',  type=str, required=True, nargs='+', help='datasets')        
	parser.add_argument('-output', type=str, required=True, help='path to store the embeddings')
	parser.add_argument('-dim', type=int, required=True, help='size of embeddings or number of topics')
	parser.add_argument('-model', choices=['skip','pv-dm','pv-dm-concat','pv-dbow','lda'], required=True, help='model')
	parser.add_argument('-epochs',   type=int, default=5, help='number of epochs')
	parser.add_argument('-workers',  type=int, default=4, help='number of workers')
	parser.add_argument('-max_sent', type=int, help='set max number of sentences to be read (per file)')
	parser.add_argument('-min_count',type=int, default=10, help='words ocurring less than ''min_count'' times are discarded')	
	parser.add_argument('-negative_samples', type=int, default=10, help='number of negative samples for Skip-Gram training. If set to 0 then Hierarchical Softmax will be used')
	parser.add_argument('-pretrained_vecs', type=str, default=None, help='path to pre-trained word vectors to train paragraph vectors')
	return parser


if __name__ == "__main__":	
	cmdline_parser = get_parser()
	args = cmdline_parser.parse_args() 			
	print "** Induce Text Representations with Gensim **"
	print "[input > %s | max_sent: %s | workers: %d | output@%s]\n" % (repr(args.input), repr(args.max_sent), args.workers,  args.output)	

	#create output folder if it does not exist
	out_folder = os.path.dirname(args.output)
	if not os.path.exists(out_folder):		
		os.makedirs(out_folder)
		print "[created output folder: %s]" % out_folder		

	if args.model =="lda":
		train_lda(args)
	elif args.model in ["pv-dm","pv-dbow","pv-dm-concat"]:
		train_paragraph2vec(args)
	elif args.model == "skip":
		train_skipgram(args)
	else:		
		raise NotImplementedError, "unknown model: %s" % args.model
