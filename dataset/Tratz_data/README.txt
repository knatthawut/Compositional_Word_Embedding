This is the noun-compound classification dataset from:

Stephen Tratz. 2011. Semantically-enriched parsing for natural language understanding. University of Southern California.

The dataset is available here: https://www.isi.edu/publications/licensed-sw/fanseparser/index.html, provided under the Apache License 2.0. 

The dataset comes in two versions - fine-grained (37 classes) and coarse-grained (12 classes). It is split into train, validation, and test in the following ways:

- random: Noun compounds randomly split into train, validation, and test sets.
- lexical_full: Noun compounds split lexically such that each head and each modifier only appears in one of the train, validation, or test set.
- lexical_head: Noun compounds split lexically such that each head only appears in one of the train, validation, or test set.
- lexical_mod: Noun compounds split lexically such that each modifier only appears in one of the train, validation, or test set.

Each split contains the sets, and a classes.txt file containing all the class labels.
