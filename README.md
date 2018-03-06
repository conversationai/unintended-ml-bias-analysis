# Unintended Bias Analysis

Tools and resources to help analyze and ameliorate unintended bias in text
classification models, as well as datasets for evaluating and mitigating
unintended bias.

This work is part of the [Conversation AI](https://conversationai.github.io/)
project, a collaborative research effort exploring ML as a tool for better
discussions online.

Relevant Links:
 * [Our overview of unintended bias in machine learning models](https://conversationai.github.io/bias.html)
 * [Unintended bias analysis dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) - includes crowd-worker demographic information, and toxicity ratings to support analysis of potential demographic-correlated unintended bias.
 * [Ai.withthebest.com April 29, 2017 Keynote Talk on ML Fairness](https://github.com/conversationai/conversationai-bias-analysis/blob/master/presentations/AI-with-the-best%20fairness%20presentation.pdf)
 * [Wikimedia Research Showcase - November 2017: Conversation Corpora, Emotional Robots, and Battles with Bias](https://www.youtube.com/watch?v=nMENRAkeHnQ)

## Training toxicity models

We provide notebooks to train CNN based models to detect toxicity in online
comments. The notebook `src/Train Toxicity Model.ipynb` provides instructions
on how to train models using the [Unintended bias analysis
dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973).
The notebook `src/Evaluate Model.ipynb` provides an example of evaluating the
performance of pre-trained models on an arbitrary dataset.

These notebooks are written for Python 2.7. To run them:

1. Install the requirements with
```
pip install -r requirements.txt
```

2. Download the data from the [Unintended bias analysis dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) to the `data/` subdirectory.
```
curl -L https://ndownloader.figshare.com/files/7394542 -o data/toxicity_annotated_comments.tsv
curl -L https://ndownloader.figshare.com/files/7394539 -o data/toxicity_annotated.tsv
```

3. Download and extract the [GloVe embeddings](http://nlp.stanford.edu/data/glove.6B.zip) in the `data` subdirectory.
```
curl -L http://nlp.stanford.edu/data/glove.6B.zip -o data/glove.6B.zip
zip -x data/glove.6B.zip -d data/glove.6B
```

Please note that if using a virtual environment, it may be necessary to
manually set your `PYTHONPATH` environment variable in the shell to the correct
version of python for the environment.

4. Now you can open and evaluate `src/Train Toxicity Model.ipynb`:
```
jupyter notebook src/Train\ Toxicity\ Model.ipynb
```

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

## Dataset de-biasing

TODO(jetpack): add dataset of additional examples as TSV with columns rev\_id,
comment, and split. also add tool to recreate dataset from wikipedia dump.

TODO(nthain,jetpack): upload new model trained on new dataset.

This technique mitigates the dataset bias found in the **Dataset bias
evaluation** section by determining the deficit of non-toxic comments for each
of the bias terms split by comment lengths. This deficit is then addressed by
sampling presumed non-toxic examples from Wikipedia articles.

These new examples can then be added to the original dataset (to all splits,
training as well as test). This allows (1) training a new model on the
augmented, de-biased dataset and (2) evaluating the new and old models on the
augmented test set in addition to the original test set.

## Qualitative model comparison

TODO(jetpack): add tools and some screenshots.

These tools provide qualitative comparisons of model results on a given
dataset. We find these tools useful for understanding the behavior of similar
models, such as when testing different de-biasing techniques.

The confusion matrix diff tool shows tables of the largest score changes for
the same examples, segmented according to the different sections of the
confusion matrix. It shows the "new" false positives/negatives and true
positives/negatives that one would get if upgrading from one model to the
other.

The score scatterplot tool plots the model's scores as a scatterplot. Each
point represents an example, the original model's score is the `x` position and
the new model's score is the `y` position. The points are colored according to
the true label. For similar models, most points should fall close to the `y=x`
line. Points that are far from that line are examples with larger score
differences between the two models.

If the new model on the y-axis is a proposed update to the model on the x-axis,
then one would hope to see mostly positive labels in the upper left corner (new
true positives) and mostly negative labels in the bottom right corner (new true
negatives).

## Data Description

The `Prep Wikipedia Data.ipynb` notebook will generate the following datasets where
SPLIT indicates whether the data is in the train, test, or dev splits:

wiki_SPLIT.csv: The original Wikipedia data from the Figshare dataset, processed and split.
wiki_debias_SPLIT.csv: The above data which is additionally augmented with Wikipedia article
comments to debias on a set of terms (see `Dataset_bias_analysis.ipynb` for details).
wiki_debias_random_SPLIT.csv: The wiki_SPLIT.csv augmented with a random selection of Wikipedia article
columns that are of roughly the same length as those used to augment wiki_debias_SPLIT.csv. This is
used as a control in our experiments.
