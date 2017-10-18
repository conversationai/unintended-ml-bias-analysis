"""Analysis of model bias.

We look at differences in model scores as a way to compare bias in different
models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_tool import ToxModel, compute_auc

MODEL_DIR = '../models/'
ORIG_MADLIBS_PATH = '../eval_datasets/bias_madlibs_89k.csv'
SCORED_MADLIBS_PATH = '../eval_datasets/bias_madlibs_89k_scored.csv'
MADLIBS_TERMS_PATH = 'bias_madlibs_data/adjectives_people.txt'

### Model scoring

# Scoring these dataset for dozens of models actually takes non-trivial amounts
# of time, so we save the results as a CSV. The resulting CSV includes all the
# columns of the original dataset, and in addition has columns for each model,
# containing the model's scores.

def postprocess_madlibs(madlibs):
    """Modifies madlibs data to have standard 'text' and 'label' columns."""
    # Native madlibs data uses 'Label' column with values 'BAD' and 'NOT_BAD'.
    # Replace with a bool.
    madlibs['label'] = madlibs['Label'] == 'BAD'
    madlibs.drop('Label', axis=1, inplace=True)
    madlibs.rename(columns={'Text': 'text'}, inplace=True)

def score_dataset(df, models, text_col):
    """Scores the dataset with each model and adds the scores as new columns."""
    for model in models:
        name = model.get_model_name()
        print('{} Scoring with {}...'.format(datetime.datetime.now(), name))
        df[name] = model.predict(df[text_col])

def load_scored_madlibs(models, scored_path=SCORED_MADLIBS_PATH,
                        orig_path=ORIG_MADLIBS_PATH):
    if os.path.exists(scored_path):
        print('Using previously scored data:', scored_path)
        return pd.read_csv(scored_path)

    madlibs = pd.read_csv(orig_path)
    postprocess_madlibs(madlibs)
    score_dataset(madlibs, models, 'text')
    print('Saving scores to:', scored_path)
    madlibs.to_csv(scored_path)
    return madlibs


### Model score analysis: AUC

def model_family_auc(dataset, model_names, label_col):
    aucs = [compute_auc(dataset[label_col], dataset[model_name])
            for model_name in model_names]
    return {
        'aucs': aucs,
        'mean': np.mean(aucs),
        'median': np.median(aucs),
        'std': np.std(aucs),
    }

def plot_model_family_auc(dataset, model_names, label_col, min_auc=0.9):
    result = model_family_auc(dataset, model_names, label_col)
    print('mean AUC:', result['mean'])
    print('median:', result['median'])
    print('stddev:', result['std'])
    plt.hist(result['aucs'])
    plt.gca().set_xlim([min_auc, 1.0])
    plt.show()
    return result

## Per-term AUC analysis.

def read_madlibs_terms():
    with open(MADLIBS_TERMS_PATH) as f:
        return [term.strip() for term in f.readlines()]

def balanced_term_subset(df, term, text_col):
    """Returns data subset containing term balanced with sample of other data.

    We draw a random sample from the dataset of other examples because we don't
    care about the model's ability to distinguish toxic from non-toxic just
    within the term-specific dataset, but rather its ability to distinguish for
    the term-specific subset within the context of a larger distribution of
    data.
    """
    term_df = df[df[text_col].str.contains(r'\b{}\b'.format(term), case=False)]
    nonterm_df = df[~df.index.isin(term_df.index)].sample(len(term_df))
    combined = pd.concat([term_df, nonterm_df])
    return combined

def model_family_name(model_names):
    """Given a list of model names, returns the common prefix."""
    prefix = os.path.commonprefix(model_names)
    if not prefix:
        raise ValueError("couldn't determine family name from model names")
    return prefix.strip('_')

def per_term_aucs(dataset, terms, model_families, text_col, label_col):
    """Computes per-term 'pinned' AUC scores for each model family."""
    records = []
    for term in terms:
        term_subset = balanced_term_subset(dataset, term, text_col)
        term_record = {'term': term, 'subset_size': len(term_subset)}
        for model_family in model_families:
            family_name = model_family_name(model_family)
            aucs = [compute_auc(term_subset[label_col], term_subset[model_name])
                    for model_name in model_family]
            term_record.update({
                family_name + '_mean': np.mean(aucs),
                family_name + '_median': np.median(aucs),
                family_name + '_std': np.std(aucs),
                family_name + '_aucs': aucs,
            })
        records.append(term_record)
    return pd.DataFrame(records)


### Model score analysis: confusion rates

def confusion_matrix_counts(df, score_col, label_col, threshold):
    return {
        'tp': len(df[(df[score_col] >= threshold) & (df[label_col] == True)]),
        'tn': len(df[(df[score_col] < threshold) & (df[label_col] == False)]),
        'fp': len(df[(df[score_col] >= threshold) & (df[label_col] == False)]),
        'fn': len(df[(df[score_col] < threshold) & (df[label_col] == True)]),
    }
