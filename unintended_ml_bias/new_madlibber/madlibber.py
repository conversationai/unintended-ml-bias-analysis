import re
import csv

from enums import Toxicity, WordType, WordConnotation, WordGender

class Madlibber(object):
  def __init__(self, path_helper):
    self.path_helper = path_helper
    self.__words = {}

  def load_and_sanity_check_words(self):
    f = open(self.path_helper.word_file, 'r')
    csv_f = csv.reader(f)
    next(csv_f)  # skip header
    count = 0
    for line in csv_f:
      word_type, word_conn, word_gender, word = line

      if word == '':
        continue

      if not self.__sanity_check(word_type, WordType):
        raise ValueError("'{}' is not a valid word type".format(word_type))

      if not self.__sanity_check(word_conn, WordConnotation):
        raise ValueError("'{}' is not a valid word connotation".format(word_connotation))

      if not self.__sanity_check(word_gender, WordGender):
        raise ValueError("'{}' is not a valid word gender".format(word_gender))

      unicode_word = word.decode('utf-8')
      word_key = self.get_word_key(word_type, word_conn, word_gender)
      self.__words.setdefault(word_key, [])
      self.__words[word_key].append(unicode_word)
      count += 1
    f.close()

    print("Loaded {} words".format(count))
    print("Loaded {} word keys".format(len(self.__words.keys())))
    print("Loaded word keys: {}".format(", ".join(["'{}'".format(k) for k in self.__words.keys()])))

  def fill_templates(self):
    f = open(self.path_helper.sentence_template_file, 'r')
    csv_f = csv.reader(f)
    next(csv_f)  # skip header

    fout = open(self.path_helper.output_file, 'w')
    csv_fout = csv.writer(fout)
    csv_fout.writerow(['template','toxicity','phrase'])

    count = 0
    for line in csv_f:
      template_count = 0
      template, toxicity, phrase = line
      print("Working on template '{}' with toxicity '{}' and phrase '{}'".format(template, toxicity, phrase))

      if phrase == '':
        continue

      if not self.__sanity_check(toxicity, Toxicity):
        raise ValueError("'{}' is not a valid toxicity".format(toxicity))

      required_words = re.findall('\{(.*?)\}',phrase)
      if len(required_words) == 0:
        raise ValueError("Template '{}' does not require fill-in: '{}'".format(template, phrase))
      for r in required_words:
        if r not in self.__words:
          raise ValueError("'{}' is not a valid type of fill-in".format(r))

      unicode_phrase = phrase.decode('utf-8')
      for words in self.__iterate_words(required_words, 0):
        output_phrase = unicode_phrase.format(**words)
        csv_fout.writerow([template, toxicity, output_phrase.encode('utf-8')])
        count += 1
        template_count += 1

      print("Output {} sentences for template '{}', toxicity '{}'".format(template_count, template, toxicity))
    f.close()
    fout.close()

    print("Output {} total sentences".format(count))

  def get_word_key(self, word_type, word_connotation, word_gender):
    return "{}_{}_{}".format(word_type, word_connotation, word_gender)

  def __sanity_check(self, item, enum):
    return enum.exists(item)

  def __iterate_words(self, required_words, required_words_index):
    is_last = required_words_index == (len(required_words) - 1)
    required_words_key = required_words[required_words_index]
    for word in self.__words[required_words_key]:
      if is_last:
        yield {required_words_key: word}
      else:
        for words in self.__iterate_words(required_words, required_words_index+1):
          words[required_words_key] = word
          yield words
