The gold standard parallel corpus can be loaded with:

```
from classic_alignment import *
gold_bsc = BisegmentationCorpus.create_from_pickle('gold_standard_bisegmentation_corpus.pickle')
```

The files from `sentence_segmenter_train.zip`, to be found in the Materials directory of the repository, must be unpacked into a subdirectory called `sentence segmenter train` of the current working directory. While building up the bisegmentation corpus object the function above will be expecting the sentence-segmented text files to be present in `sentence segmeter train/segmented/`.
