The gold standard parallel corpus can be loaded with:

```
from classic_alignment import *
gold_bsc = BisegmentationCorpus.create_from_pickle('gold_standard_bisegmentation_corpus.pickle')
```

For this to work, the files from `sentence_segmenter_train.zip`, which can be found in the Materials directory of this repository, must be unpacked into a subdirectory called `sentence segmenter train` of the current working directory. While building up the bisegmentation corpus object the function above will be expecting the sentence-segmented text files to be present in `sentence segmenter train/segmented/`.
