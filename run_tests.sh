#!/bin/bash
set -e
set -x

pushd unintended_ml_bias
python model_bias_analysis_test.py
popd
