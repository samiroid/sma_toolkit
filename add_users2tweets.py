### SCRIPT TO TRANSFORM THE CORPUS BY INSERTING THE USERNAME IN THE CENTER OF THE TWEETS

from collections import Counter
from pdb import set_trace
import argparse
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert usernames into their twwets")
    parser.add_argument('corpus_in', type=str, help='input corpus')        
    parser.add_argument('corpus_out', type=str, help='output corpus')    
    parser.add_argument('-window_size', type=int, required=True, help="window size")
    args = parser.parse_args()

print "[converting %s -> %s | window: %d]" % (args.corpus_in,args.corpus_out,args.window_size)

with open(args.corpus_out,"w") as fod:
	with open(args.corpus_in) as fid:
		z=0
		for x in fid:
			u,t = x.split("\t")
			t = t.split()
			l = len(t)
			nt = []
			for i in xrange(len(t)):
				if not i%(args.window_size-1):
					nt.append(u)
				nt.append(t[i])				
			if not z%1000:
				sys.stdout.write("\r  > %d" % z)
				sys.stdout.flush()
			z+=1
		fod.write(" ".join(nt)+"\n")


