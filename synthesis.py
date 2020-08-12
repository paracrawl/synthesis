#!env python3

from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim import utils

import os
import argparse
import codecs
from collections import Counter
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus(object):
  """An interator that yields sentences (lists of str)."""

  def __init__(self, corpusfile):
    self.corpusfile = corpusfile

  def __iter__(self):
    corpus_path = datapath(self.corpusfile)
    for line in open(corpus_path):
      # assume there's one document per line, tokens separated by whitespace
      #yield line.split(" ") # keeps everything
      yield list(utils.tokenize(line, lowercase=False)) # removes numbers, punc
      #yield utils.simple_preprocess(line, min_len=1, max_len=25) # also removes too small and too big words, lowercases

# train a word embedding model
def train(corpusfile, modelfile):
  sentences = MyCorpus(corpusfile)
  model = Word2Vec(sentences, size=512, window=5, min_count=10, workers=20)
  model.save(modelfile)
  return model

# compute word alignment with fast align
def align_parallel_corpus(corpusfile_f, corpusfile_e):
  fast_align = os.path.dirname(__file__) + "/fast_align/build/"
  os.system("paste " + corpusfile_f + " " + corpusfile_e + " " +
            "> " + outdir + "/fast-align-input")
  os.system(fast_align + "fast_align -i " + outdir + "/fast-align-input d -v -o " +
            "> " + outdir + "/fast-align")
  os.system(fast_align + "fast_align -i " + outdir + "/fast-align-input d -v -o -r " + 
            "> " + outdir + "/fast-align-inverse")
  os.system(fast_align + "atools -c grow-diag-final-and " + 
            "-i " + outdir + "/fast-align " + 
            "-j " + outdir + "/fast-align-inverse " +
            "> " + outdir + "/aligned")
  return outdir + "/aligned"

# get word translation pairs that are relatively probable
def get_possible_replacement_word_pairs(corpusfile_f, corpusfile_e, alignmentfile):
  fh_f = codecs.open(corpusfile_f, "r", encoding='utf-8')
  fh_e = codecs.open(corpusfile_e, "r", encoding='utf-8')
  fh_a = codecs.open(alignmentfile, "r", encoding='utf-8')

  count_e = Counter()
  count_f = Counter()
  translation = {}
  while(True):
    # read a sentence pair with word alignment
    line_a = fh_a.readline().strip()
    if len(line_a) == 0:
      break
    word_f = fh_f.readline().strip().split(" ")
    word_e = fh_e.readline().strip().split(" ")

    # increase word counts
    for e in word_e:
      count_e[ e ] += 1
    for f in word_f:
      count_f[ f ] += 1

    # how often is each word aligned?
    aligned_e = Counter()
    aligned_f = Counter()
    for alignment_point in line_a.split(" "):
      fi, ei = [ int(x) for x in alignment_point.split("-") ]
      aligned_e[ ei ] += 1
      aligned_f[ fi ] += 1
 
    # record translations for 1-1 alignments
    for alignment_point in line_a.split(" "):
      fi, ei = [ int(x) for x in alignment_point.split("-") ]
      if aligned_e[ ei ] > 1 or aligned_f[ fi ] > 1:
        continue
      f, e = word_f[fi], word_e[ei]
      if f not in translation:
        translation[ f ] = Counter()
      translation[ f ][ e ] += 1
  fh_e.close()
  fh_f.close()
  fh_a.close()
    
  # filter to word pairs where p(e|f)>.1 and p(f|e)>.1
  reliable_translation = {}
  for f in translation:
    if count_f[ f ] < 20:
      continue
    for e in translation[f]:
      if count_e[ e ] < 20:
        continue
      c = translation[f][e]
      if c * 10 > count_e[ e ] and c * 10 > count_f[ f ]:
        if f not in reliable_translation:
          reliable_translation[ f ] = []
        reliable_translation[ f ].append( [e, c] )
  print(reliable_translation)
  return reliable_translation

def get_similar_pairs(e, f, max_similar_words_considered, corpus_translation):
  # get list of similar words to e and f
  #f_list = model_f.most_similar(positive=[f], topn=max_similar_words_considered, case_insensitive=False)
  #e_list = model_e.most_similar(positive=[e], topn=max_similar_words_considered, case_insensitive=False)
  f_list = model_f.most_similar(positive=[f], topn=max_similar_words_considered)
  e_list = model_e.most_similar(positive=[e], topn=max_similar_words_considered)
  e_index = {}
  for item in e_list:
    e_index[ item[0] ] = item[1]

  # see there are pairs between the two lists that match glossary
  match_type = 'no similar in f lex'
  similar = []
  total_count = 0
  # loop through all words similar to f
  for item in f_list:
    f_similar = item[0]
    f_similar_score = item[1]

    # f_similar has to be in glossary
    if f_similar in corpus_translation:
      if match_type is not "ok":
        match_type = 'no translation in e lex'
      #print(e,f,corpus_translation[ f_similar ],
      #print('\ttranslation for',f_similar)
      for f_similar_translation in corpus_translation[ f_similar ]:
        f_similar_translation_word = f_similar_translation[0]
        f_similar_translation_count = f_similar_translation[1]
        #print('\t\t\tpossible translation', f_similar_translation)

        # f_similar's translation has to be in list of similar words
        if f_similar_translation_word in e_index:
          e_similar_score = e_index[f_similar_translation_word]
          match_type = 'ok'
          #print(f, e, f_similar, f_similar_translation_word, f_similar_score, e_similar_score, f_similar_translation_count)
          item = { "f": f_similar, 
                   "e": f_similar_translation_word, 
                   "similarity": f_similar_score * e_similar_score,
                   "count": int(f_similar_translation_count) }
          total_count += int(f_similar_translation_count)
          similar.append(item)

  # prune list if too big
  if total_count <= MAX_SENTENCE_PAIRS:
    return match_type, total_count, similar
  else:
    new_total_count = 0
    new_similar = []
    for item in sorted(similar, key = lambda i: -i['similarity']):
      new_similar.append(item)
      new_total_count = new_total_count + item['count']
      if new_total_count >= MAX_SENTENCE_PAIRS:
        break
    return 'pruned', new_total_count, new_similar
    

# get index of how words are translated in the corpus
# (only reliable 1-1 translations)
def load_possible_replacement_word_pairs( lexfile ):
  corpus_translation = {}
  fh = codecs.open(lexfile, "r", encoding='utf-8')
  for line in fh:
    word = line.strip().split('\t')
    f = word[0]
    e = word[1]
    count = word[4]
    if f not in corpus_translation:
      corpus_translation[f] = []
    corpus_translation[f].append([e,count])
  fh.close()
  return corpus_translation

# find replacement candidates (f,e) for glossary translations
def get_replacement_pairs(glossaryfile, model_f, model_e, corpus_translation):
  similar_list = []
  fh = codecs.open(glossaryfile, "r", encoding='utf-8')
  for line in fh:
    # get glossary item (f,e)
    word = line.strip().split('\t')
    f = word[0]
    e = word[1]
    #print(e,f)

    if not(e in model_e and f in model_f):
      print(f, e, '\t', 'no embeddings')
      continue

    # cast an increasingly wider net over similar words
    max_similar_words_considered = int(MAX_SIMILAR_WORDS_CONSIDERED / (2**6))
    while True:
      comment, count, similar = get_similar_pairs(e, f, max_similar_words_considered, corpus_translation)
      if count >= MAX_SENTENCE_PAIRS or max_similar_words_considered == MAX_SIMILAR_WORDS_CONSIDERED:
        print(f, e, '\t', comment, '\t', max_similar_words_considered, '\t', count, '\t', similar)
        similar_list.append( [f, e, similar] )
        break
      max_similar_words_considered = 2 * max_similar_words_considered
      if max_similar_words_considered > MAX_SIMILAR_WORDS_CONSIDERED:
        print(f, e, '\t', comment, '\t', max_similar_words_considered, '\t', 0)
        break
  fh.close()

  # re-index by (f,e) pairs in corpus
  replacement = {}
  for similar in similar_list:
    f = similar[0]
    e = similar[1]
    for item in similar[2]:
      f_corpus = item['f']
      e_corpus = item['e']
      similarity = item['similarity']
      glossary = { "f":f, "e":e, "similarity": similarity }
      if f_corpus not in replacement:
        replacement[ f_corpus ] = {}
      if e_corpus not in replacement[ f_corpus ]:
        replacement[ f_corpus ][ e_corpus ] = []
      replacement[ f_corpus ][ e_corpus ].append(glossary)

  return replacement

# generate new sentence pairs
def generate_new_sentence_pairs(corpus_f, corpus_e, alignmentfile, outfile, replacement):
  fh_f = codecs.open(corpus_f, "r", encoding='utf-8')
  fh_e = codecs.open(corpus_e, "r", encoding='utf-8')
  fh_a = codecs.open(alignmentfile, "r", encoding='utf-8')
  fh_out = codecs.open(outfile, "w", encoding='utf-8')
  for sentence_f in fh_f:
    sentence_f = sentence_f.strip()
    sentence_e = fh_e.readline().strip()
    f = sentence_f.split(' ')
    e = sentence_e.split(' ')

    # loop through aligned words
    slot = []
    for item in fh_a.readline().strip().split(' '):
      fi, ei = item.split('-')
      f_word = f[int(fi)]
      e_word = e[int(ei)]
      # is this corpus word pair replaceable with a glossary pair?
      if f_word in replacement and e_word in replacement[f_word]:
        slot_item = {"fi": int(fi), 
                     "ei": int(ei),
                     "replacement": replacement[f_word][e_word]}
        slot.append( slot_item )

    # if replacement slots found, loop through replacements
    if len(slot)>0:
      #print(sentence_f, sentence_e, slot)
      slot_index = [0] * len(slot)
      max = 1
      for item in slot:
        max = max * len(item["replacement"])
      for ignore in range(min(max, 100)):
      
        # carry out replacement
        #print("===", slot_index, slot)
        f_new = f.copy()
        e_new = e.copy()
        item_list = []
        for i in range(len(slot)):
          item = slot[i]        # word pair to be changed
          index = slot_index[i]  # replacement option
          r = item['replacement'][ index ]
          f_new_word = r['f']
          e_new_word = r['e']
          f_new[ item['fi'] ] = f_new_word
          e_new[ item['ei'] ] = e_new_word
          item_list.append(f_new_word + " " + e_new_word + " " + str(r['similarity']))
          # slot_index[i]['similarity']
        fh_out.write("\t".join([", ".join(item_list), sentence_f, sentence_e, " ".join(f_new), " ".join(e_new)]) + "\n")
        #print(sentence_f)
        #print(sentence_e)
        #print(" ".join(f_new)) 
        #print(" ".join(e_new)) 

        # increase counter
        i = len(slot)-1
        while i>=0:
          slot_index[i] = slot_index[i] + 1
          if slot_index[i] < len(slot[i]["replacement"]):
            continue
          slot_index[i] = 0
          i = i-1
  fh_f.close()
  fh_e.close()
  fh_a.close()

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True,
                    help="working directory to store experimental data")
parser.add_argument("--glossary", required=True,
                    help="text file with glossary terms, tab-separate source/target word")
parser.add_argument("--monolingual-corpus", nargs=2,
                    help="monolingual corpora, source and target file")
parser.add_argument("--embedding", nargs=2,
                    help="word embeddings for source and target language")
parser.add_argument("--parallel-corpus", nargs=2, required=True,
                    help="parallel corpus, source and target file")
parser.add_argument("--alignment",
                    help="word alignments for parallel corpus, if they pre-computed")
args = parser.parse_args()

MAX_SIMILAR_WORDS_CONSIDERED = 10240
MAX_SENTENCE_PAIRS = 200

# check if files exist
file_not_found = False
for file in [ args.alignment, args.glossary, args.monolingual_corpus, args.parallel_corpus, args.embedding ]:
  if file is None:
    continue
  if type(file) is list:
    for individual_file in file:
      if not os.path.exists(individual_file):
        print("ERROR: file does not exist: " + individual_file)
  else:
    if not os.path.exists(file):
      print("ERROR: file does not exist: " + file)
if file_not_found:
  exit(1)

# create working directory
outdir = args.dir
if not os.path.exists(outdir):
    os.makedirs(outdir)

# train word embedding models on parallel data augmented
# with monolingual data that contains glossary terms
if args.embedding is not None:
  modelfile_f, modelfile_e = args.embedding
  model_f = Word2Vec.load( modelfile_f )
  model_e = Word2Vec.load( modelfile_e )
else:
  if args.monolingual_corpus is None:
    print("ERROR: You need to specify monolingual corpora (needed to train word embeddings) with --monolingual-corpus, or alternatively pretrained word embedding files with --embedding")
    exit(1)
  mono_corpus_f, mono_corpus_e = args.monolingual_corpus
  model_f = train( os.path.abspath(mono_corpus_f), outdir + "/embedding.f" )
  model_e = train( os.path.abspath(mono_corpus_e), outdir + "/embedding.e" )
  
# compute word alignment for parallel corpus (using fast align)
corpus_f, corpus_e = args.parallel_corpus
if args.alignment is not None:
  alignmentfile = args.alignment
else:
  alignmentfile = align_parallel_corpus(corpus_f, corpus_e)

# main processing steps
glossaryfile = args.glossary
corpus_translation = get_possible_replacement_word_pairs(corpus_f, corpus_e, alignmentfile)
replacement = get_replacement_pairs(glossaryfile, model_f, model_e, corpus_translation)
generate_new_sentence_pairs(corpus_f, corpus_e, alignmentfile, outdir + "/synthetic-corpus", replacement)
