# MScThesis
Program code for my CS master's thesis on using cross-lingual lexical vector space models for bitext alignment

The repository contains most of the code that was used in writing the Case Study chapter of my MSc thesis on using vector space models for the alignment of parallel texts, so-called bitexts.

These cover the following steps of the process:
* pre-processing of the OJ documents, especially extracting plain natural language text from the xml documents
* removal of useless documents (including foreign-language documents, ones where two parallel texts are mostly identical for some reason, too short documents, etc.)
* sentence-level segmentation of documents, i.e. sentence-boundary detection
* analysis of various statistical features of documents in the corpus
* management of segments, segmentations, bisegments, bisegmentations and parallel corpora
* Gale & Church (1992)-style classic length-based alignment and its extension that takes into consideration numbers and numberings appearing in the bitext to be aligned
* extension of the classic algorithm by anchor segments
* managements of monolingual and bilingual corpora used for training vector space lexical models, including management of vocabularies with frequencies
* normalisation of tokens (lower-casing, lemmatisation)
* implementation of a standard monolingual lexical vector space model using the skip-gram with negative sampling (SGNS) algorithm
* implementation of a version of a fastText-style variant of the SGNS algorithm that includes purely n-gram-based subword information
* implementation of an alternative variant of the previous algorithm that uses subword information in a smarter way, based on Hafer & Weiss (1974)-style morpheme boundary heuristics
* implementation of a bilingual version of the SGNS algorithm that is trained bidirectionally on bisegments
* benchmarks to compare different vector space models with regard to their ability to differentiate between segments that should and ones that should not be linked in course of the alignment process

The code was written in Python and is largely undocumented and uncommented. Some parts, notably statistical calculations and benchmarks, are presented as iPython notebooks.
