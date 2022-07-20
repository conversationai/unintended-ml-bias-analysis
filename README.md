# Unintended ML Bias Analysis

This repository contains the Sentence Templates datasets we use to evaluate and
mitigate unintended machine learning bias in [Perspective
API](https://perspectiveapi.com). See our accompanying [blog post](https://medium.com/jigsaw/identifying-machine-learning-bias-with-updated-data-sets-7c36d6063a2c) to learn more about how we created these datasets.

This work is part of the [Conversation AI](https://conversationai.github.io/)
project, a collaborative research effort exploring ML as a tool for better
discussions online.

**NOTE: We moved outdated scripts, notebooks, and other resources to the
[archive](archive/) subdirectory. We no longer maintain those resources, but you
may find some of the content helpful. In particular, see
[model_bias_analysis.py](archive/unintended_ml_bias/model_bias_analysis.py) for
an example of how to analyze model bias.**

## Background

As part of the Perspective API model training process, we evaluate identity-term
bias in our models on synthetically generated and “templated” test sets. To
generate these sets, we plug in identity terms into both toxic and non-toxic
template sentences. For example, given templates like “I am a \<modifier>
\<identity>”, we evaluate differences in score on sentences like:

> “I am a kind American"
>
> “I am a kind Muslim"

Scores that vary significantly may indicate identity term bias within the model.

For more reading on unintended bias and how we measure bias using the resulting
model scores, see:

- Our [overview](https://conversationai.github.io/bias.html) of unintended bias
  in machine learning models
- Our [Measuring and Mitigating Unintended Bias in Text
  Classification](https://research.google/pubs/pub46743/) paper for a deeper
  dive into this approach for mitigating unintended bias
- Our [Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
  Classification](https://research.google/pubs/pub48094/) paper for details on
  the various metrics we use to measure unintended bias
- Our [model
  cards](https://developers.perspectiveapi.com/s/about-the-api-model-cards) for
  an overview of our model training process and model performance metrics
- [Model Cards for Model Reporting](https://research.google/pubs/pub48120/) for
  an introduction into model cards

# Usage

We encourage researchers and developers to use these datasets to test for biases
in their own models. However, Sentence Templates alone are insufficient for
eliminating identity bias in machine learning language models. The examples are
simple and unlikely to appear in real-world data and may reflect our own biases.
The identity terms also vary across languages because direct word-for-word
translation of identity terms across languages is not sufficient, or even
possible, given differences in cultures, religions, idioms, and identities.

# Copyright and license

All code in this repository is made available under the Apache 2 license. All
data in this repository is made available under the Creative Commons Attribution
4.0 International license (CC By 4.0). A full copy of the license can be found
at https://creativecommons.org/licenses/by/4.0/
