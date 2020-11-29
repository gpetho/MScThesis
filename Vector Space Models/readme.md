This directory contains the Python implementation of four versions of the skip-gram with negative sampling algorithm which was introduced in word2vec to generate distributional vector space representations of words:
* a simple monolingual implementation very similar to the one in word2vec, but in Python, more flexible and slower
* a cross-lingual version of this implementation which is trained using a sentence-aligned parallel corpus
* a version of the monolingual implementation that uses subword information in generating word vectors, like fastText
* a cross-lingual version of the latter algorithm which takes subwords into consideration in both languages
