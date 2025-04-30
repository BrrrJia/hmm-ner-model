# hmm-ner-model

Implement a HMM-based NER.

1. Read the corpus and convert the text sequence into IOB-format.
2. Compute the model parameters from the training data, including initial probabilities, observation probabilities(print the top 10 most possible token for each label), and transition probabilities.
3. Identify the 10 instances with the highest gold probability from the validation corpus.
4. Predict the IOB-labels of the sequence in the validation data using Viterbi Algorithm. #TODO

## Data:

NER-de-train.tsv: training data, from GermNER corpus(https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germaner.html)

NER-de-dev.tsv: validation data, from GermNER corpus(https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germaner.html)

## Packages used:

numpy, pandas, csv, collections, more_itertools
