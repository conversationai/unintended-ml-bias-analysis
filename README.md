# Unintended Bias Analysis

Tools and resources to help analyze and ameliorate unintended bias in text
classification models, as well as datasets evaluation and mitigating unintended bias.
This work is part of the [Conversation AI](https://conversationai.github.io/) effort, a collaborative
research effort exploring ML as a tool for better discussions online.

Relevant Links:
 * [Our overview of unintended bias in machine learning models](https://conversationai.github.io/bias.html)
 * [Unintended bias analysis dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) - includes crowd-worker demographic information, and toxicity ratings to support analysis of potential demographic-correlated unintended bias.
 * [Ai.withthebest.com April 29, 2017 Keynote Talk on ML Fairness](https://github.com/conversationai/conversationai-bias-analysis/blob/master/AI-with-the-best%20fairness%20presentation.pdf)


## Training toxicity models

We provide notebooks to train CNN based models to detect toxicity in online comments. The notebook `src/Train Toxicity Model.ipynb` provides instructions on how to train models using the [Unintended bias analysis dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973). The notebook `src/Evaluate Model.ipynb` provides an example of evaluating the performance of pre-trained models on an arbitrary dataset.

These notebooks are written for Python 2.7 and in order to run them you must first:

1. Install the requirements with 
```
pip install -r requirements.txt
```
2. (optional: to skip training) Download the latest [model](https://storage.googleapis.com/unintended-ml-bias-analysis/models/wiki_tox_labels_v1_model.h5) and [tokenizer](https://storage.googleapis.com/unintended-ml-bias-analysis/models/wiki_tox_labels_v1_tokenizer.pkl) to the `models/` subdirectory.
3. (optional: only if training) Download the data from the [Unintended bias analysis dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) to the `data/` subdirectory.
4. (optional: only if training) Download and extract the [GloVe embeddings](http://nlp.stanford.edu/data/glove.6B.zip) in the `data` subdirectory.

Please note that if using a virtual environment, it may be necessary to manually set your PYTHONPATH environment variable in the shell to the correct version of python for the environment.

## Dataset bias evaluation

TODO(jetpack): add notes, screenshots for the dataset bias analysis tool

## Model bias evaluation

### "Bias madlibs" eval dataset

This dataset is one tool in evaluating our de-biasing efforts. For a given
template, a large difference in model scores when single words are substituted
may point to a bias problem. For example, if "I am a gay man" gets a much
higher score than "I am a tall man", this may indicate bias in the model.

The madlibs dataset contains 89k examples generated from templates and word
lists. The dataset is `eval_datasets/bias_madlibs_89k.csv`, a CSV consisting of
2 columns.  The generated text is in `Text`, and the label is `Label`, either
`BAD` or `NOT_BAD`.

The script (`src/bias_madlibs.py`) and word lists (`src/bias_madlibs_data/`)
used to generate the data are also included.

TODO(jetpack): add notes about future work / improvements.

### Fuzzed test set

This technique involves modifying a test set
by ["fuzzing"](https://en.wikipedia.org/wiki/Fuzzing) over a set of identity
terms in order to evaluate a model for bias.

Given a test set and a set of terms, we replace all instances of each term in
the test data with a random other term from the set. The idea is that the
specific term used for each example should not be the key feature in determining
the label for the example. For example, the sentence "I had a <x> friend growing
up" should be considered non-toxic, and "All <x> people must be wiped off the
earth" should be considered toxic for all values of `x` in the terms set.

The code in `src/Bias_fuzzed_test_set.ipynb` reads the Wikipedia Toxicity
dataset and builds an identity-term-focused test set. It writes unmodified and
fuzzed versions of that test set. One can then evaluate a model on both test
sets. Doing significantly worse on the fuzzed version may indicate a bias in the
model. The datasets are `eval_datasets/toxicity_fuzzed_testset.csv` and
`eval_datasets/toxicity_nonfuzzed_testset.csv`. Each CSV consists of 3 columns:
ID unedr `rev_id`, the comment text under `comment`, and the True/False label
under `toxic`.

This is similar to the bias madlibs technique, but has the advantage of using
more realistic data. One can also use the model's performance on the original
vs. fuzzed test set as a bias metric.
