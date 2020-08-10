# synthesis

INSTALLATION

You will need fast_align and Moses.

pip3 install gensim cython CuPy

git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../..

USAGE
python ./synthesis.py --dir my-result-dir \
    --monolingual-corpus any-text-with-glossary-words.de \
                         any-text-with-glossary-words.en \
    --parallel-corpus existing-parallel-corpus.de \
                      existing-parallel-corpus.en \
    --glossary terminology >& log

BUG
* full path required for mono corpus
* check early if input files exist
* prepare toy corpus
