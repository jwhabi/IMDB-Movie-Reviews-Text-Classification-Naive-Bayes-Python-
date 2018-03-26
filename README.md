# IMDB-Review-Text-Classification-Naive-Bayes-Python-
****Goal:**** Implementation and Evaluation of the Naive Bayes algorithm in Python for Movie reviews sourced from IMDB

****Process Overview:**** 
-First curate the reviews in the training dataset to create a bag of words dictionary with each word having a unique 
 identifier and frequency based on which probabilities will be calculated. (Call to IMDB.py and Vocab.py)
-Then read the test dataset and obtain labels for each review by lookinng at the bag of words from the training dataset 
 and their probabilites using the frequency tables as mentioned above. (NaiveBayes.py)
-All probabilities are calculated using logsum. (NaiveBayes.py)
-An alpha level of smoothing is also set to offset for newly occuring words.
-Evaluation occurs by measuring accuracy,precision and recall for different levels of alpha and plotting 
 the precision vs recall curve.   (Eval.py)
-Additionally, modules for printing Features (top 20 influential words in each class) and a predict function to give the 
 prediction for any review are also implemented. (NaiveBayes.py)


****Dataset:**** 
Overview

This dataset contains movie reviews along with their associated binary
sentiment polarity labels. It is intended to serve as a benchmark for
sentiment classification. This document outlines how the dataset was
gathered, and how to use the files provided. 

Dataset 

The core dataset contains 50,000 reviews split evenly into 25k train
and 25k test sets. The overall distribution of labels is balanced (25k
pos and 25k neg). We also include an additional 50,000 unlabeled
documents for unsupervised learning. 

Files

There are two top-level directories [train/, test/] corresponding to
the training and test sets. Each contains [pos/, neg/] directories for
the reviews with binary labels positive and negative. Within these
directories, reviews are stored in text files named following the
convention [[id]_[rating].txt] where [id] is a unique id and [rating] is
the star rating for that review on a 1-10 scale. For example, the file
[test/pos/200_8.txt] is the text for a positive-labeled test set
example with unique id 200 and star rating 8/10 from IMDb. The
[train/unsup/] directory has 0 for all ratings because the ratings are
omitted for this portion of the dataset.

We also include the IMDb URLs for each review in a separate
[urls_[pos, neg, unsup].txt] file. A review with unique id 200 will
have its URL on line 200 of this file. Due the ever-changing IMDb, we
are unable to link directly to the review, but only to the movie's
review page.

****Citations:****


@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  
                Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
Andrew Maas
amaas@cs.stanford.edu

References

Potts, Christopher. 2011. On the negativity of negation. In Nan Li and
David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20,
636-659.



