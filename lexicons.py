import argparse 
import codecs
import os
from pdb import set_trace
### LEXICON PARSERS 

def parse_NRCEmolex(path_in, path_out):    
    emolex_polarity = {}
    emolex_sadness = {}
    emolex_joy = {}
    emolex_trust= {}
    emolex_anticipation = {}
    emolex_anger = {}
    emolex_disgust = {}
    emolex_surprise = {}
    emolex_fear = {}    
    fd = codecs.open(path_in, "r","utf-8")
    i=0    
    for line in fd:        
        line = line.replace(u"\ufeff","").replace(u"\x00","").replace(u"\ufb01","fi").replace(u"\ufb02","fl").replace("\n","")        
        split_aux = line.split()    
        i+=1        
        if len(split_aux) != 11:
            # print "skiped: %s (%d)" % (split_aux,len(split_aux))            
            continue
        word = split_aux[0]
        try:
            pos   = int(split_aux[1].replace("positive:",""))
            neg   = int(split_aux[2].replace("negative:",""))
            ang   = int(split_aux[3].replace("anger:",""))
            ant   = int(split_aux[4].replace("anticipation:",""))
            disg  = int(split_aux[5].replace("disgust:",""))
            fear  = int(split_aux[6].replace("fear:",""))
            joy   = int(split_aux[7].replace("joy:",""))
            sad   = int(split_aux[8].replace("sadness:",""))
            surp  = int(split_aux[9].replace("surprise:",""))
            trust = int(split_aux[10].replace("trust:",""))
            
            #polarity lexicon
            if pos == neg:
                emolex_polarity[word] = 0
            elif pos == 1:
                emolex_polarity[word] = 1
            elif neg == 1:
                emolex_polarity[word] = -1            
            #anger
            if ang == 1:
                emolex_anger[word] = 1
            else:
                emolex_anger[word] = 0
            #anticipation
            if ant == 1:
                emolex_anticipation[word] = 1
            else:
                emolex_anticipation[word] = 0
            #disgust
            if disg == 1:
                emolex_disgust[word] = 1
            else:
                emolex_disgust[word] = 0
            #fear
            if fear == 1:
                emolex_fear[word] = 1
            else:
                emolex_fear[word] = 0
            #joy
            if joy == 1:
                emolex_joy[word] = 1
            else:
                emolex_joy[word] = 0
            #sadness
            if sad == 1:
                emolex_sadness[word] = 1
            else:
                emolex_sadness[word] = 0
            #surprise
            if surp == 1:
                emolex_surprise[word] = 1
            else:
                emolex_surprise[word] = 0
            #trust
            if trust == 1:
                emolex_trust[word] = 1
            else:
                emolex_trust[word] = 0

        except ValueError:
            continue     
    fd.close()

    emotions = ["polarity","anger","anticipation","disgust","fear","joy",
    			"sadness","surprise","trust"]
    lexicons = [emolex_polarity, emolex_anger, emolex_anticipation, emolex_disgust, 
    			emolex_fear, emolex_joy, emolex_sadness, emolex_surprise, emolex_trust]    

    for emo, lex in zip(emotions, lexicons):
    	out_file = path_out + emo + ".txt"        
    	save_lexicon(lex, out_file)

def parse_opinion_mining_lex(path_in, path_out):    
    sentiment_lexicon = {}
    
    # import positive-words, negative-words
    positive_lexicon = codecs.open(path_in + "positive-words.txt", "r")
    negative_lexicon = codecs.open(path_in + "negative-words.txt", "r")
    for word in positive_lexicon:
        # ignore comments
        if word.startswith(";"):
            continue
        word = word.replace("\n", "")
        sentiment_lexicon[word] = 1
    positive_lexicon.close()

    for word in negative_lexicon:
        # ignore comments
        if word.startswith(";"):
            continue
        word = word.replace("\n", "")
        sentiment_lexicon[word] = 0
    negative_lexicon.close()

    save_lexicon(sentiment_lexicon, path_out)


def parse_mpqa(path_in, polarity_out=None, subjectivity_out=None):   
    
    assert polarity_out is not None or subjectivity_out is not None

    with open(path_in) as fid:        
        polarity_lex = {}       
        subjectivity_lex = {}
        for l in fid:           
            datum = l.split()
            word = datum[2].replace("word1=","")            
            subjectivity_lex[word] = 0 if datum[0]=='type=weaksubj' else 1
            if datum[5] == 'priorpolarity=negative':
                pol = -1
            elif datum[5] == 'priorpolarity=neutral':
                pol = 0
            elif datum[5] == 'priorpolarity=positive':
                pol = 1         
            polarity_lex[word] = pol            
    
    if polarity_out is not None:
        save_lexicon(polarity_lex, polarity_out)

    if subjectivity_out is not None:
        save_lexicon(subjectivity_lex, subjectivity_out)

### LOAD AND SAVE LEXICONS
def save_lexicon(lex, path_out):
    #if path_out folder does not exist, create it
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

    with codecs.open(path_out,"w","utf-8") as fod:
        sorted_lex = sorted(lex.items(),key=lambda x:x[1],reverse=True)
        for wrd, lab in sorted_lex:
            fod.write(u"%s\t%s\n" % (wrd,str(lab)))

def parse_lex(path_in, path_out, word_pos=0, label_pos=1, sep='\t',skip_first=False):
    lex = read_lex(path_in,word_pos,label_pos,sep,skip_first)
    save_lexicon(lex, path_out)    

def read_lex(path_in, word_pos=0, label_pos=1, sep='\t',skip_first=False):
    """
    	Read a lexicon
    	word_pos: index of the word
    	label_pos: index of the label
    	sep: separator token
    	skip_first: if True, ignore the first line
    """
    ignored=[]
    with open(path_in) as fid:
        lex = {}
        if skip_first:
            fid.readline()
        for l in fid:
            try:
                datum = l.split(sep)
                lex[datum[word_pos]] = float(datum[label_pos])
            except:
                ignored.append(l)
        print "ignored ", ignored
        #print "".join(ignored)
        
	return lex


def get_parser():
	parser = argparse.ArgumentParser(description="Parse Lexicons")
	parser.add_argument('-in',  type=str, required=True, nargs='+', help='path to the lexicon')        
	parser.add_argument('-out', type=str, required=True, help='output folder')
	parser.add_argument('-lex', choices=['mpqa','emolex','oml','duyu','anew','anew-II'], required=True, help='model')
	
	return parser


if __name__ == "__main__":	
	cmdline_parser = get_parser()
	args = cmdline_parser.parse_args()

	if args.lex == 'mpqa':
		pass
	elif args.lex == '':
		pass
	elif args.lex == '':
		pass
	elif args.lex == '':			
		pass
	elif args.lex == '':
		pass
