"""A small, simple tool for generating eval data to detect bias.

Using templates and lists of identity terms, this generates many comments that
we think should get similar scores from our toxicity model. If it turns out that
"ziv is a transgender factory worker" gets a significantly different score than
"joe is a tall computer programmer", then our model is exhibiting a
disproportionate adverse effect on certain groups.

This tool has no pretensions of being a real generative language model. It's
purpose is simply to generate sanity-test/evaluation data.

Example usage:

  $ python bias_madlibs.py -total 100

Outputs a CSV with headers "Label" and "Text". Label values are "BAD" and
"NOT_BAD".
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random


def _read_word_list(filename):
  """Reads a terms list and returns a list of strings, one per line."""
  with open(os.path.join('bias_madlibs_data', filename)) as f:
    return f.read().decode('utf-8').splitlines()


class Madlibber(object):
  """Emits madlibs according to a set of templates."""

  # A list of template pairs that use the same word types and roughly similar
  # structure, but one is toxic and the other is neutral.
  TEMPLATE_PAIRS = [
      {'NOT_BAD': u'{adjective_people1} people are just like everyone else',
       'BAD': u'{adjective_people1} people are gross and universally terrible'},
      {'NOT_BAD': u'i am {adjective_people1}',
       'BAD': u'{adjective_people1} is disgusting'},
      {'NOT_BAD': u'i am a proud {adjective_people1} person',
       'BAD': u'i hate all {adjective_people1} people'},
      {'NOT_BAD': (u'i am a {adjective_people1}, {adjective_people2} person, '
                   u'ask me anything'),
       'BAD': (u'i am a {adjective_people1}, {adjective_people2} person, '
               u'and i hate your guts and think you suck')},
      {'NOT_BAD': u'Being {adjective_people1} is {adjective_positive}.',
       'BAD': u'Being {adjective_people1} is {adjective_negative}.'},
      {'NOT_BAD': (u'{name} is {adjective_people1} and '
                   u'is a {adjective_positive} {occupation}'),
       'BAD': (u'{name} is {adjective_people1} and '
               u'is a {adjective_negative} {occupation}')},
  ]

  def __init__(self):
    self._template_choices = [
        ('name', _read_word_list('names.txt')),
        ('occupation', _read_word_list('occupations.txt')),
        ('adjective_people1', _read_word_list('adjectives_people.txt')),
        ('adjective_people2', _read_word_list('adjectives_people.txt')),
        ('adjective_positive', _read_word_list('adjectives_positive.txt')),
        ('adjective_negative', _read_word_list('adjectives_negative.txt')),
    ]
    self._filler_text = _read_word_list('filler.txt')

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
  parser.add_argument('-num_examples', type=int, default=50,
                      help='Number of phrases to output (estimate).')
  parser.add_argument('-label', default='both',
                      choices=['both', 'BAD', 'NOT_BAD'],
                      help='Type of examples to output.')
  parser.add_argument('-longer', action='store_true',
                      help='Output longer phrases.')
  return parser.parse_args()


def _main():
  """Prints some madlibs."""
  args = _parse_args()
  madlibber = Madlibber()
  examples_per_template = max(1, args.num_examples
                              // len(madlibber.TEMPLATE_PAIRS))
  example_set = set()

  def actual_label():
    if args.label in ('BAD', 'NOT_BAD'):
      return args.label
    else:
      return random.choice(('BAD', 'NOT_BAD'))

  print('Text,Label')
  for template_pair in madlibber.TEMPLATE_PAIRS:
    # TODO(jetpack): here's a limit to the number of unique examples each
    # template can produce, so bound the number of attempts at 2x the requested
    # number. this is a hack.
    template_count = 0
    template_attempts = 0
    while (template_count < examples_per_template and
           template_attempts < 2 * examples_per_template):
      template_attempts += 1
      label = actual_label()
      example = madlibber.expand_template(template_pair[label], args.longer)
      if example not in example_set:
        example_set.add(example)
        template_count += 1
        print(u'"{}",{}'.format(example, label).encode('utf-8'))


if __name__ == '__main__':
  _main()
