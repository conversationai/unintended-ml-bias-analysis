import os

class PathHelper(object):
  def __init__(self, word_file, sentence_template_file, output_file):
    if not os.path.exists(word_file):
      raise IOError("Input word file '{}' does not exist!".format(word_file))
    if not os.path.isfile(word_file):
      raise IOError("Input word file '{}' is not a file!".format(word_file))
    self.word_file = word_file

    if not os.path.exists(sentence_template_file):
      raise IOError("Input sentence template file '{}' does not exist!".format(sentence_template_file))
    if not os.path.isfile(sentence_template_file):
      raise IOError("Input sentence template  file '{}' is not a file!".format(sentence_template_file))
    self.sentence_template_file = sentence_template_file

    if os.path.basename(output_file) == '':
      raise IOError("Output file '{}' cannot be a directory.".format(output_file))
    output_dirname = os.path.dirname(output_file)
    if not os.path.exists(output_dirname):
      print("Output directory '{}' does not exist...creating".format(output_dirname))
      os.makedirs(output_dirname)
    self.output_file = output_file
