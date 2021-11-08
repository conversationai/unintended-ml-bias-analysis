# Unintended ML Bias Analysis

This repository contains the Sentence Templates datasets we use to evaluate and
mitigate unintended machine learning bias in [Perspective
API](https://perspectiveapi.com). See our accompanying [blog post]() for more
information

This work is part of the [Conversation AI](https://conversationai.github.io/)
project, a collaborative research effort exploring ML as a tool for better
discussions online.

## Background

As part of the Perspective API model training process, we evaluate identity-term
bias in our models on synthetically generated and “templated” test sets where a
range of identity terms are swapped into both toxic and non-toxic template
sentences. For example, given templates like “I am a proud [identity] person”,
we evaluate differences in score on sentences like:

> “I am a proud Latino person”
>
> “I am a proud gay person”
>
> “I am a proud Muslim person”

_Note that this evaluation looks at only the identity terms present in the
text._

For more reading on unintended bias and how we measure bias using the resulting
model scores, see:

- Our [overview](https://conversationai.github.io/bias.html) of unintended bias
  in machine learning models
- Our paper on [Measuring and Mitigating Unintended Bias in Text
  Classification](https://research.google/pubs/pub46743/) for a deeper dive into
  this approach and the metrics we use
- Our [model
  cards](https://developers.perspectiveapi.com/s/about-the-api-model-cards) for
  an overview of our model training process and model performance metrics

# Usage

Sentence Templates is not itself sufficient for eliminating identity bias in
machine learning language models. The examples are simple and unlikely to appear
in real-world data, and both the templates and word-lists necessarily reflect
our own biases. Direct word-for-word translation of sentences and identity terms
across languages is further not sufficient, or even possible, given differences
in cultures, religions, idioms, and identities.
