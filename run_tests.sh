#!/bin/bash
set -e
set -x

pushd unintended_ml_bias
python model_bias_analysis_test.py
popd

# IPython "Tests" execute notebooks
jupyter-nbconvert --to notebook --execute \
  presentations/FAT_Star_Tutorial_Measuring_Unintended_Bias_in_Text_Classification_Models_with_Real_Data.ipynb
