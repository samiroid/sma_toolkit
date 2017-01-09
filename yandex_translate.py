import argparse
from yandex_translate import YandexTranslate
import sys

def translate_corpus(api_key, translation_pair,
                     path_in, path_out, max_sent=float('inf')):
    
    translator = YandexTranslate(api_key)
    fails = []
    with open(path_out,"w") as fod:    
        with open(path_in) as fid:
            for i, l in enumerate(fid):
                if i > max_sent:
                    break
                elif not i%1000:
                    sys.stdout.write("\r> %d" % i)
                    sys.stdout.flush()        
                try:        
                    tr = translator.translate(l,translation_pair)['text'][0].strip("\n")
                except: 
                    fails.append(l)
                fod.write(tr.encode("utf-8")+"\n")
    print "\ntranslated corpus @ %s " % path_out
    print "fails"
    print fails



parser = argparse.ArgumentParser(description="Preprocess corpus")
parser.add_argument('corpus_in', type=str, help='input corpus')        
parser.add_argument('corpus_out', type=str, help='output (preprocessed) corpus')            
parser.add_argument('-pair', required=True, choices=["en-es","es-en"], help='translation pair')
parser.add_argument('-api_key', required=True, type=str, help='Yandex Translation API key')
parser.add_argument('-max_sent', type=int, help='max number of sentences to be proces')

args = parser.parse_args()
if args.max_sent:
	translate_corpus(args.api_key, args.pair, 
					 args.corpus_in, args.corpus_out,  
					 max_sent=args.max_sent)
else:
	translate_corpus(args.api_key, args.pair, 
					 args.corpus_in, args.corpus_out)
