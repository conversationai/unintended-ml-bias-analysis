# Unintended ML Bias Analysis

This repository contains the Sentence Templates datasets we use to evaluate and
mitigate unintended machine learning bias in [Perspective
API](https://perspectiveapi.com). See our accompanying [blog post]() to learn
more about how we created these datasets.

This work is part of the [Conversation AI](https://conversationai.github.io/)
project, a collaborative research effort exploring ML as a tool for better
discussions online.

## Background

As part of the Perspective API model training process, we evaluate identity-term
bias in our models on synthetically generated and “templated” test sets. To
generate these sets, we plug in identity terms into both toxic and non-toxic
template sentences. For example, given templates like “I am a <modififier>
<identity>”, we evaluate differences in score on sentences like:

> “I am a kind American"
>
> “I am a kind Muslim"

Scores that vary significantly may indicate identity term bias within the model.

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

We encourage researchers and developers to use these datasets to test for biases
in their own models. However, Sentence Templates alone are insufficient for
eliminating identity bias in machine learning language models. The examples are
simple and unlikely to appear in real-world data and may reflect our own biases.
The identity terms also vary across languages because direct word-for-word
translation of identity terms across languages is not sufficient, or even
possible, given differences in cultures, religions, idioms, and identities.

# License

TODO: Add this in
