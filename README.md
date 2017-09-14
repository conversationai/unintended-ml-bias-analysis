# Unintended Bias Analysis

Tools and resources to help analyze and ameliorate unintended bias in text
classification models, as well as datasets evaluation and mitigating unintended bias.
This work is part of the [Conversation AI](https://conversationai.github.io/) effort, a collaborative
research effort exploring ML as a tool for better discussions online.

Relevant Links:
 * [Our overview of unintended bias in machine learning models](https://conversationai.github.io/bias.html)
 * [Unintended bais analysis dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) - includes crowd-worker demographic information, and toxicity ratings to support analysis of potential demographic-correlated unintended bias.
 * [Ai.withthebest.com April 29, 2017 Keynote Talk on ML Fairness](https://github.com/conversationai/conversationai-bias-analysis/blob/master/AI-with-the-best%20fairness%20presentation.pdf)


## "Bias madlibs" eval dataset

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
