import argparse
import cPickle
from ipdb import set_trace
from __init__ import word_2_idx, stratified_sampling, shuffle_split
import numpy as np
import os
import embeddings as emb_utils

def get_parser():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument('-tr', type=str, required=True, help='train file')
    parser.add_argument('-ts', type=str, required=True, nargs='+', help='test file(s)')        
    parser.add_argument('-max_inst', type=int, help='maximum number of instances')    
    parser.add_argument('-rand_seed', type=int, default=1234, help='random seed')    
    parser.add_argument('-split_perc', type=float, help='split the training data')    
    parser.add_argument('-out', type=str, required=True, help='features file')
    parser.add_argument('-bow', action="store_true", help='BOW features')
    parser.add_argument('-boe', type=str, help='path the embeddings (extract BOE features)')    
    parser.add_argument('-nlse', type=str, help='path the embeddings (extract NLSE features)')    
    return parser

def features(msgs, labels, wrd2idx, lbl2idx, max_inst=None, split_perc=None):
	"""
		Compute word features
		msgs: messages
		labels: labels
		wrd2idx: dictionary mapping words to indices
		lbl2dix: dictionary mapping classes to numeric values
		max_inst: If set, specifies a maximum number of (stratified) samples
		split_perc: If set, specifies splits the data
	"""
	X = [ np.array([wrd2idx[w] for w in m.split() if w in wrd2idx]) for m in msgs ]	
	Y = np.array([lbl2idx[l] for l in labels])	
	#Keep only max instances
	if max_inst is not None:
		print "[max instances: %d]" % max_inst				
		sample = stratified_sampling(zip(X, Y), max_inst)
		X = [x[0] for x in sample]
		Y = [x[1] for x in sample]
	#split into train/dev
	if split_perc is not None:
		print "[split: tr:%.2f/ts:%.2f]" % (split_perc,(1-split_perc))
		train, dev = shuffle_split(zip(X, Y), split_perc=split_perc)		
		X     = [x[0] for x in train]
		Y     = [x[1] for x in train]
		X_dev = [x[0] for x in dev]
		Y_dev = [x[1] for x in dev]
		return X, Y, X_dev, Y_dev		
	else:
		return X, Y

if __name__=="__main__":		
	parser = get_parser()
	args = parser.parse_args()    			
	train_file = args.tr	
	train_msgs = []	
	train_lbls = []	
	test_msgs  = []	
	print "[reading data]"
	# read training data
	with open(train_file,"r") as fid:
		for l in fid:
			# print l
			splt = l.split("\t")
			train_lbls.append(splt[0])
			train_msgs.append(splt[1])	
	# convert to numeric labels
	lbl2idx = word_2_idx(train_lbls)		
	#shuffle the training data
	rng   = np.random.RandomState(args.rand_seed)
	shuff = rng.randint(0,len(train_msgs),len(train_msgs)) 
	train_lbls = np.array(train_lbls)[shuff]
	train_msgs = np.array(train_msgs)[shuff]
	test_sets = []	
	for test_file in args.ts:
		msgs = []
		lbls = []
		print test_file			
		with open(test_file) as fid:
			for l in fid:
				splt = l.split("\t")
				if splt[0] not in lbl2idx: continue
				lbls.append(splt[0])
				msgs.append(splt[1])	
				test_msgs.append(splt[1])
		test_sets.append([test_file,msgs,lbls])			
	#output folder
	if not os.path.isdir(os.path.dirname(args.out)): os.makedirs(os.path.dirname(args.out))	
	#feature extraction
	if args.bow:	
		###### BOW feature extraction					
		wrd2idx = word_2_idx(train_msgs)			
		print "[saving BOW features]"
		#train data				
		if args.split_perc is not None:			
			X_train, Y_train, X_dev, Y_dev = features(train_msgs, train_lbls, 
													  wrd2idx, lbl2idx, 
													  max_inst=args.max_inst, 
													  split_perc=args.split_perc)
			#save dev file
			dev_file = args.out + "BOW_" + os.path.splitext(os.path.split(train_file)[1])[0] + "_dev.pkl"
			with open(dev_file,"wb") as fod: cPickle.dump([wrd2idx, X_dev, Y_dev],fod,-1)
			print "\t> Extracted %s -> %s" % (train_file, dev_file)
		else:			
			X_train, Y_train = features(train_msgs, train_lbls, wrd2idx, lbl2idx, max_inst=args.max_inst)
		#save
		out_file = args.out + "BOW_" + os.path.splitext(os.path.split(train_file)[1])[0] + ".pkl"
		with open(out_file,"wb") as fod: cPickle.dump([wrd2idx, X_train, Y_train],fod,-1)		
		print "\t> Extracted %s -> %s" % (train_file, out_file)		
		#test data
		for test_file, test_msgs, test_lbls in test_sets:									
			X_test, Y_test = features(test_msgs, test_lbls, wrd2idx, lbl2idx)
			#save
			out_file = args.out + "BOW_" + os.path.splitext(os.path.split(test_file)[1])[0] + ".pkl"
			with open(out_file,"wb") as fod: cPickle.dump([wrd2idx, X_test, Y_test],fod,-1)		
			print "\t> Extracted %s -> %s" % (test_file, out_file)
	if args.boe is not None:
		###### BOE feature extraction				
		# when using embeddings it is ok to include words from the test data in the vocabulary 
		# (all the embeddings are available at inference time)
		assert os.path.exists(args.boe), "couldn't find the embeddings (check the path)" 
		print "[saving BOE features]"				
		wrd2idx = word_2_idx(np.concatenate((train_msgs,test_msgs)))		
		if args.split_perc is not None:			
			X_train, Y_train, X_dev, Y_dev = features(train_msgs, train_lbls, 
													  wrd2idx, lbl2idx, 
													  max_inst=args.max_inst, 
													  split_perc=args.split_perc)
			#save dev file
			dev_file = args.out + "BOE_" + os.path.splitext(os.path.split(train_file)[1])[0] + "_dev.pkl"
			with open(dev_file,"wb") as fod: cPickle.dump([wrd2idx, X_dev, Y_dev],fod,-1)
			print "\t> Extracted %s -> %s" % (train_file, dev_file)
		else:			
			X_train, Y_train = features(train_msgs, train_lbls, wrd2idx, lbl2idx, max_inst=args.max_inst)
		#save
		out_file = args.out + "BOE_" + os.path.splitext(os.path.split(train_file)[1])[0] + ".pkl"
		with open(out_file,"wb") as fod: cPickle.dump([wrd2idx, X_train, Y_train],fod,-1)		
		print "\t> Extracted %s -> %s" % (train_file, out_file)		
		#test data
		for test_file, test_msgs, test_lbls in test_sets:									
			X_test, Y_test = features(test_msgs, test_lbls, wrd2idx, lbl2idx)
			#save
			out_file = args.out + "BOE_" + os.path.splitext(os.path.split(test_file)[1])[0] + ".pkl"
			with open(out_file,"wb") as fod: cPickle.dump([wrd2idx, X_test, Y_test],fod,-1)		
			print "\t> Extracted %s -> %s" % (test_file, out_file)		
		print "[loading word embeddings: %s]" % args.boe
		E = emb_utils.get_embeddings(args.boe, wrd2idx)			
		out_file = args.out+"/E_"+ os.path.splitext(os.path.split(train_file)[1])[0] +".pkl"
		#save embedding matrix
		with open(out_file,"w") as fod: cPickle.dump(E, fod, -1)	
	if args.nlse is not None:
		###### NLSE feature extraction		
		# when using embeddings it is ok to include words from the test data in the vocabulary 
		# (all the embeddings are available at inference time)
		#make sure the embedding file exists
		assert os.path.exists(args.nlse), "couldn't find the embeddings (check the path)" 
		wrd2idx = word_2_idx(np.concatenate((train_msgs,test_msgs)))
		print "[saving NLSE features]"
		#train data
		if args.split_perc is not None:			
			X_train, Y_train, X_dev, Y_dev = features(train_msgs, train_lbls, 
													  wrd2idx, lbl2idx, 
													  max_inst=args.max_inst, 
													  split_perc=args.split_perc)
			#save dev file
			dev_file = args.out + "NLSE_" + os.path.splitext(os.path.split(train_file)[1])[0] + "_dev.pkl"
			with open(dev_file,"wb") as fod: cPickle.dump([wrd2idx, X_dev, Y_dev],fod,-1)
			print "\t> Extracted %s -> %s" % (train_file, dev_file)
		else:			
			X_train, Y_train = features(train_msgs, train_lbls, wrd2idx, lbl2idx, max_inst=args.max_inst)
		# Concatenate train data into a single numpy array, keep start and end
		# indices
		lens = np.array([len(tr) for tr in X_train]).astype(int)
		st = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype(int)
		ed = (st + lens)
		x = np.zeros((ed[-1], 1))
		for i, ins_x in enumerate(X_train):	x[st[i]:ed[i]] = np.array(ins_x,dtype=int)[:, None]
		X_train = x     
		Y_train = np.array(Y_train)[:, None] # Otherwise slices are scalars not Tensors
		# save
		out_file = args.out + "NLSE_" + os.path.splitext(os.path.split(train_file)[1])[0] + ".pkl"
		with open(out_file,"w") as fod: cPickle.dump([X_train, Y_train, st, ed], fod, cPickle.HIGHEST_PROTOCOL)		
		print "\t> Extracted %s -> %s" % (train_file, out_file)
			#test data
		for test_file, test_msgs, test_lbls in test_sets:
			X_test, Y_test = features(test_msgs, test_lbls, wrd2idx, lbl2idx)
			out_file = args.out + "NLSE_" + os.path.splitext(os.path.split(test_file)[1])[0] + ".pkl"
			with open(out_file,"w") as fod: cPickle.dump([X_test, Y_test], fod, -1)		
			print "\t> Extracted %s -> %s" % (test_file, out_file)
		print "[loading word embeddings: %s]" % args.nlse
		E = emb_utils.get_embeddings(args.nlse, wrd2idx)			
		out_file = args.out+"/E_"+ os.path.splitext(os.path.split(train_file)[1])[0] +".pkl"
		#save embedding matrix
		with open(out_file,"w") as fod: cPickle.dump(E, fod, -1)	