import argparse

from path_helper import PathHelper
from madlibber import Madlibber

def parse_args():
  """Returns parsed arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-input_words',
      type=str,
      required=True,
      help='The input words to substitute into templates.')
  parser.add_argument(
      '-input_sentence_templates',
      type=str,
      required=True,
      help='The input sentence templates.')
  parser.add_argument(
      '-output_file',
      type=str,
      required=True,
      help='The output file of filled in templates.')
  return parser.parse_args()

def main():
  args = parse_args()
  ph = PathHelper(args.input_words, args.input_sentence_templates, args.output_file)
  m = Madlibber(ph)
  m.load_and_sanity_check_words()
  m.fill_templates()

if __name__ == '__main__':
  main()
