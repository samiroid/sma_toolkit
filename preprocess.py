import argparse
from ext import twokenize
from ext.tweetokenize import Tokenizer
import numpy as np
import re
import sys

# emoticon regex taken from Christopher Potts' script at http://sentiment.christopherpotts.net/tokenizing.html
emoticon_regex = r"""(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)"""

twk = Tokenizer(ignorequotes=False,usernames=False,urls=False,numbers=False)

def rescale_features(x, axis, mode='std'):
    assert mode in ['std','unit'], "Unknown mode: accepted modes are 'std' and 'unit'"
    new_x = None
    if mode == 'std':
        new_x = (x - np.mean(x,axis=axis))/np.std(x,axis=axis)              
    elif mode == 'unit':
        new_x = x/np.sqrt(np.sum(x**2,axis=axis))              
    return new_x

def rescale_torange(X,old_min,old_max,new_min,new_max):
    w = (new_max-new_min)/(old_max-old_min)
    new_X = ((X-old_max)*w)+new_max
    return new_X

def max_reps(sentence, n=3):

    """
        Normalizes a string to at most n repetitions of the same character
        e.g, for n=3 and "helllloooooo" -> "helllooo"
    """
    new_sentence = ''
    last_c = ''
    max_counter = n
    for c in sentence:
        if c != last_c:
            new_sentence+=c
            last_c = c
            max_counter = n
        else:
            if max_counter > 1:
                new_sentence+=c
                max_counter-=1
            else:
                pass
    return new_sentence

def count_emoticon_polarity(message):
    """
        returns the number of positive, neutral and negative emoticons in message
    """
    emoticon_list = re.findall(emoticon_regex, message)
    polarity_list = []
    for emoticon in emoticon_list:
        if emoticon in ['8:', '::', 'p:']:
            continue # these are false positives: '8:48', 'http:', etc
        polarity = emoticon_polarity(emoticon)
        polarity_list.append(polarity)          
    emoticons = Counter(polarity_list)
    pos = emoticons[1]
    neu = emoticons[0]
    neg = emoticons[-1]
    
    return pos,neu,neg

def remove_emoticons(message):
    return re.sub(emoticon_regex,'',message)

def emoticon_polarity(emoticon):
    
    eyes_symbol = re.findall(r'[:;=8]', emoticon) # find eyes position    
    #if valid eyes are not found return 0
    if len(eyes_symbol) == 1:
        eyes_symbol = eyes_symbol[0]    
    else:
        return 0
    mouth_symbol = re.findall(r'[\)\]\(\[dDcCpP/\}\{@\|\\]', emoticon) # find mouth position    
    #if a valid mouth is not found return 0
    if len(mouth_symbol) == 1:
        mouth_symbol = mouth_symbol[0]
    else:
        return 0
    eyes_index = emoticon.index(eyes_symbol)
    mouth_index = emoticon.index(mouth_symbol)
    # this assumes typical smileys like :)
    if mouth_symbol in [')', ']', '}', 'D', 'd']:
        polarity = +1
    elif mouth_symbol in ['(', '[', '{', 'C', 'c']:
        polarity = -1
    elif mouth_symbol in ['p', 'P', '\\', '/', ':', '@', '|']:
        polarity = 0
    else:
        raise Exception                
    # now we reverse polarity for reversed smileys like (:
    if eyes_index > mouth_index:
        polarity = -polarity

    return polarity

def preprocess(m, sep_emoji=False):
    assert type(m) == unicode
    
    m = m.lower()    
    m = max_reps(m)
    #replace user mentions with token '@user'
    user_regex = r".?@.+?( |$)|<@mention>"    
    m = re.sub(user_regex," @user ", m, flags=re.I)
    #replace urls with token 'url'
    m = re.sub(twokenize.url," url ", m, flags=re.I)        
    tokenized_msg = ' '.join(twokenize.tokenize(m)).strip()
    if sep_emoji:
        #tokenize emoji, this tokenzier however has a problem where repeated punctuation gets separated e.g. "blah blah!!!"" -> ['blah','blah','!!!'], instead of ['blah','blah','!','!','!']
        m_toks = tokenized_msg.split()
        n_toks = twk.tokenize(tokenized_msg)         
        if len(n_toks)!=len(m_toks):
            #check if there is any punctuation in this string
            has_punct = map(lambda x:x in twk.punctuation, n_toks)
            if any(has_punct):  
                new_m = n_toks[0]
                for i in xrange(1,len(n_toks)):
                    #while the same punctuation token shows up, concatenate
                    if has_punct[i] and has_punct[i-1] and (n_toks[i] == n_toks[i-1]):
                        new_m += n_toks[i]
                    else:
                        #otherwise add space
                        new_m += " "+n_toks[i]                   
                tokenized_msg = new_m                
    return tokenized_msg.lstrip()

def preprocess_corpus(corpus_in, corpus_out, max_sent=float('inf'), sep_emoji=False):

    with open(corpus_out,"w") as fod:    
        with open(corpus_in) as fid:
            for i, l in enumerate(fid):
                if i > max_sent:
                    break
                elif not i%1000:
                    sys.stdout.write("\ri:%d" % i)
                    sys.stdout.flush()
                nl = preprocess(l.decode("utf-8"),sep_emoji)
                # set_trace()
                fod.write(nl.encode("utf-8")+"\n")
    print "\nprocessed corpus @ %s " % corpus_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess corpus")
    parser.add_argument('corpus_in', type=str, help='input corpus')        
    parser.add_argument('corpus_out', type=str, help='output (preprocessed) corpus')            
    parser.add_argument('-max_sent', type=int, help='max number of sentences to be proces')
    parser.add_argument('-sep_emoji', action="store_true", default=False, help='separate emojis')

    args = parser.parse_args()
    if args.max_sent:
    	preprocess_corpus(args.corpus_in, args.corpus_out, sep_emoji=args.sep_emoji, max_sent=args.max_sent)
    else:
    	preprocess_corpus(args.corpus_in, args.corpus_out, sep_emoji=args.sep_emoji)
