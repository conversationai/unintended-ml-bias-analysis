import argparse

from path_helper import PathHelper
from format_helper import FormatHelper
from template_words import TemplateWords
from template_words_helper import TemplateWordsHelper
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
  twh = TemplateWordsHelper(FormatHelper, TemplateWords)
  m = Madlibber(ph, FormatHelper, twh)
  m.load_sanity_check_templates_and_infer_word_categories()
  m.load_and_sanity_check_words()
  m.display_statistics()
  should_fill = raw_input("Do you wish to generate the sentences? [y/N]")
  if should_fill == "y":
    m.fill_templates()
  print("Done. Exiting...") 

if __name__ == '__main__':
  main()
