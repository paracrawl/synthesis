# Synthesis

This tool creates an synthetic parallel corpus for training machine translation models.
It is intended for the use case when an existing parallel corpus lacks translations for
known glossary terms with specified translations. It creates sentence pairs that contain
these glossary term translations. 

The tool takes existing sentence pairs and replaces words that are similar to the glossary
terms and their translations (that are also similar to the translations of the glossary terms).
It makes use of word embeddings to assess word similarity and word alignments computed by
fast-align to identify the translations of words in the existing parallel corpus.

The following resources are required:
* A file with glossary terms and their translations (currently only single words allowed)
* A parallel corpus
* Monolingual corpora for source and target language that contain the glossary terms and their translations, respectively

Glossary terms have to be contained in the monolingual corpora at least 10 times -- the more the better for assessing their similarity to words in the parallel corpus.

## EXAMPLE
Assume that you glossary has the following translation:
* Kranken 
* sick

The similarity model based on word embeddings may find the following similar translation pair in the corpus:
* Armen
* poor

Note that both ''Kranken'' and ''Armen'', as well as ''sick'' and ''poor'' have to be similar.

The existing parallel corpus may contain this word translation pair in the following sentence pair:
* die Armen sich nicht abgesichert .
* the poor go unprotected .

Note that the tool assumes that all data is tokenized - it does not perform any additional pre-processing.

Based on all this information, the following synthetic sentence pair is generated:
* die Kranken sich nicht abgesichert .
* the sick go unprotected .

## INSTALLATION

You will need fast_align and Moses.

```
pip3 install gensim cython CuPy
```

```
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../..
```

## USAGE

You can use the following toy data:
```
wget http://www.statmt.org/paracrawl/synthesis/synthesis-data.tgz
tar xzf synthesis-data.tgz
```

Here is the command:
```
python ./synthesis.py --dir my-result-dir \
    --monolingual-corpus `pwd`/synthesis-data/imonolingual.de \
                         `pwd`/synthesis-data/monolingual.en \
    --parallel-corpus synthesis-data/parallel.de \
                      synthesis-data/parallel.en \
    --glossary synthesis-data/terminology >& log
```
