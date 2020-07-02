import argparse

import format_helper
import madlibber
import path_helper
import word_helper


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
  ph = path_helper.PathHelper(args.input_words, args.input_sentence_templates,
                              args.output_file)
  wh = word_helper.WordHelper(format_helper.FormatHelper)
  m = madlibber.Madlibber(ph, format_helper.FormatHelper, wh)
  m.load_sanity_check_templates_and_infer_word_categories()
  m.load_and_sanity_check_words()
  m.display_statistics()
  should_fill = input('Do you wish to generate the sentences? [y/N]')
  if should_fill == 'y':
    m.fill_templates()
  print('Done. Exiting...')


if __name__ == '__main__':
  main()
