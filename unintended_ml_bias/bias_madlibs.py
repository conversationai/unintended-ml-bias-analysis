"""A small, simple tool for generating eval data to detect bias.
Using templates and lists of identity terms, this generates many comments that
we think should get similar scores from our toxicity model. If it turns out that
"ziv is a transgender factory worker" gets a significantly different score than
"joe is a tall computer programmer", then our model is exhibiting a
disproportionate adverse effect on certain groups.
This tool has no pretensions of being a real generative language model. It's
purpose is simply to generate sanity-test/evaluation data.
Example usage:
  $ python bias_madlibs.py -num_examples 100
Outputs a CSV with headers "Label", "Text", and "Template".
Label values are "BAD" and "NOT_BAD". "Template" is an ID for the template to
allow grouping results by template.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random


def _read_word_list(bias_data_dir, filename):
  """Reads a terms list and returns a list of strings, one per line."""
  with open(os.path.join(bias_data_dir, filename)) as f:
    return f.read().decode('utf-8').splitlines()


class Madlibber(object):
  """Emits madlibs according to a set of templates."""

  # A list of template pairs that use the same word types and roughly similar
  # structure, but one is toxic and the other is neutral.
  TEMPLATE_PAIRS = [
      {
          'NOT_BAD': u'{verb_positive} {adjective_people}',
          'BAD': u'{verb_negative} {adjective_people}',
          'template': 'verb_adj'
      },
      {
          'NOT_BAD': u'i am {adjective_people}',
          'BAD': u'i hate {adjective_people}',
          'template': 'am_hate_adj'
      },
      {
          'NOT_BAD': u'Being {adjective_people} is {adjective_positive}.',
          'BAD': u'Being {adjective_people} is {adjective_negative}.',
          'template': 'being_adj'
      },
      {
          'NOT_BAD': u'{name} is a {adjective_positive} {adjective_people}',
          'BAD': u'{name} is a {adjective_negative} {adjective_people}',
          'template': 'name_adj'
      },
      {
          'NOT_BAD': u'You are a {adjective_positive} {adjective_people}',
          'BAD': u'You are a {adjective_negative} {adjective_people}',
          'template': 'you_are_adj'
      },
      {
          'NOT_BAD': u'You are a {adjective_positive} {occupation}',
          'BAD': u'You are a {adjective_negative} {occupation}',
          'template': 'you_occupation'
      },
  ]

  def __init__(self, bias_data_dir):
    self._template_choices = [
        ('name', _read_word_list(bias_data_dir, 'names.txt')),
        ('occupation', _read_word_list(bias_data_dir, 'occupations.txt')),
        ('adjective_people',
         _read_word_list(bias_data_dir, 'adjectives_people.txt')),
        ('adjective_positive',
         _read_word_list(bias_data_dir, 'adjectives_positive.txt')),
        ('adjective_negative',
         _read_word_list(bias_data_dir, 'adjectives_negative.txt')),
        ('verb_positive', _read_word_list(bias_data_dir, 'verbs_positive.txt')),
        ('verb_negative', _read_word_list(bias_data_dir, 'verbs_negative.txt')),
    ]
    self._filler_text = _read_word_list(bias_data_dir, 'filler.txt')

  def expand_template(self, template, add_filler):
    """Expands the template with randomly chosen words."""
    parts = {}
    for template_key, choices in self._template_choices:
      parts[template_key] = random.choice(choices)
    expanded = template.format(**parts)
    if add_filler:
      return u'{} {}'.format(expanded, random.choice(self._filler_text))
    return expanded


def _parse_args():
  """Returns parsed arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-num_examples',
      type=int,
      default=50,
      help='Number of phrases to output (estimate).')
  parser.add_argument(
      '-bias_data_dir',
      type=str,
      default='bias_madlibs_data',
      help='Directory for bias data.')
  parser.add_argument(
      '-label',
      default='both',
      choices=['both', 'BAD', 'NOT_BAD'],
      help='Type of examples to output.')
  parser.add_argument(
      '-longer', action='store_true', help='Output longer phrases.')
  return parser.parse_args()


def _main():
  """Prints some madlibs."""
  args = _parse_args()
  madlibber = Madlibber(args.bias_data_dir)
  examples_per_template = max(
      1, args.num_examples // len(madlibber.TEMPLATE_PAIRS))
  example_set = set()

  def actual_label():
    if args.label in ('BAD', 'NOT_BAD'):
      return args.label
    else:
      return random.choice(('BAD', 'NOT_BAD'))

  print('Text,Label,Template')
  for template_pair in madlibber.TEMPLATE_PAIRS:
    # TODO(jetpack): here's a limit to the number of unique examples each
    # template can produce, so bound the number of attempts. this is a hack.
    template_count = 0
    template_attempts = 0
    while (template_count < examples_per_template and
           template_attempts < 7 * examples_per_template):
      template_attempts += 1
      label = actual_label()
      example = madlibber.expand_template(template_pair[label], args.longer)
      if example not in example_set:
        example_set.add(example)
        template_count += 1
        print(u'"{}",{},{}'.format(example, label,
                                   template_pair['template']).encode('utf-8'))


if __name__ == '__main__':
  _main()
