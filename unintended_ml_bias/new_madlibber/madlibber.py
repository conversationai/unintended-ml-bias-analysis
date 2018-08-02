import re
import csv

class Madlibber(object):
  def __init__(self, path_helper):
    self.path_helper = path_helper
    self.__templates = []
    self.__all_template_elements = set([])
    self.__toxicity = set(["toxic", "nontoxic"])
    self.__words = {}
    self.__word_types = set([])
    self.__word_connotations = set([])
    self.__word_genders = set([])

  def load_sanity_check_templates_and_infer_word_categories(self):
    print("Loading templates, sanity checking and inferring word categories...")
    f = open(self.path_helper.sentence_template_file, 'r')
    csv_f = csv.reader(f)
    next(csv_f)  # skip header

    for line in csv_f:
      template, toxicity, phrase = line
      template_elements = self.__extract_template_elements(phrase)

      if phrase == '':
        continue

      if not self.__sanity_check(toxicity, self.__toxicity):
        raise ValueError("'{}' is not a valid toxicity".format(toxicity))

      if len(template_elements) == 0:
        raise ValueError("Template '{}' does not require fill-in: '{}'".format(template, phrase))

      for t in template_elements:
        word_type, word_connotation, word_gender = self.invert_template_element(t)
        self.__word_types.add(word_type)
        self.__word_connotations.add(word_connotation)
        self.__word_genders.add(word_gender)
        self.__all_template_elements.add(t)
        
      self.__templates.append((template, toxicity, phrase, template_elements))
    f.close()

    print("Inferred word types: {}".format(" ,".join(self.__word_types)))
    print("Inferred word connotations: {}".format(" ,".join(self.__word_connotations)))
    print("Inferred word genders: {}".format(", ".join(self.__word_genders)))
    print("Done")

  def load_and_sanity_check_words(self):
    print("Loading word list...")
    f = open(self.path_helper.word_file, 'r')
    csv_f = csv.reader(f)
    next(csv_f)  # skip header

    seen_word_types = set([])
    seen_word_connotations = set([])
    seen_word_genders = set([])
    for line in csv_f:
      word_type, word_conn, word_gender, word = line

      if word == '':
        continue

      if not self.__sanity_check(word_type, self.__word_types):
        raise ValueError("'{}' is not a valid word type".format(word_type))

      if not self.__sanity_check(word_conn, self.__word_connotations):
        raise ValueError("'{}' is not a valid word connotation".format(word_conn))

      if not self.__sanity_check(word_gender, self.__word_genders):
        raise ValueError("'{}' is not a valid word gender".format(word_gender))

      seen_word_types.add(word_type)
      seen_word_connotations.add(word_conn)
      seen_word_genders.add(word_gender)

      unicode_word = word.decode('utf-8')
      template_element = self.get_template_element(word_type, word_conn, word_gender)
      if template_element not in self.__all_template_elements:
        raise ValueError("'{}' is not found in the templates".format(template_element))
      self.__words.setdefault(template_element, [])
      self.__words[template_element].append(unicode_word)
    f.close()

    unseen_word_types = self.__word_types.difference(seen_word_types)
    if len(unseen_word_types) > 0:
      raise ValueError("{} word type(s) were not found in the word list".format(", ".join(unseen_word_types)))
    unseen_word_conns = self.__word_connotations.difference(seen_word_connotations)
    if len(unseen_word_conns) > 0:
      raise ValueError("{} word connotation(s) were not found in the word list".format(", ".join(unseen_word_conns)))
    unseen_word_genders = self.__word_genders.difference(seen_word_genders)
    if len(unseen_word_conns) > 0:
      raise ValueError("{} word gender(s) were not found in the word list".format(", ".join(unseen_word_genders)))
    for t in self.__all_template_elements:
      if t not in self.__words:
        raise ValueError("Template element '{}' is not represented in the word list".format(t))

    print("Done")

  def display_statistics(self):
    print("Template statistics:")
    print("Total number of templates: {}".format(len(self.__templates)))
    print("Total number of unique template elements to fill in: {}".format(len(self.__all_template_elements)))
    print("Word statistics:")
    total = 0
    count = 1
    for k, words in self.__words.iteritems():
      n_words = len(words)
      total += n_words
      print("Template element {}: {}, Total number of words: {}".format(count, k, n_words))
      count += 1
    print("Total number of words: {}".format(total))
    total = 0
    for t in self.__templates:
      template_elements = t[-1]
      template_total = 1
      for te in template_elements:
        template_total *= len(self.__words[te])
      total += template_total
    print("Number of expected output lines: {}".format(total))

  def fill_templates(self):
    fout = open(self.path_helper.output_file, 'w')
    csv_fout = csv.writer(fout)
    csv_fout.writerow(['template','toxicity','phrase'])

    count = 0
    for template, toxicity, phrase, template_elements in self.__templates:
      template_count = 0
      print("Working on template '{}' with toxicity '{}' and phrase '{}'".format(template, toxicity, phrase))

      unicode_phrase = phrase.decode('utf-8')
      for words in self.__iterate_words(template_elements, 0):
        output_phrase = unicode_phrase.format(**words)
        csv_fout.writerow([template, toxicity, output_phrase.encode('utf-8')])
        count += 1
        template_count += 1

      print("Output {} sentences for template '{}', toxicity '{}'".format(template_count, template, toxicity))
    fout.close()

    print("Output {} total sentences".format(count))

  def get_template_element(self, word_type, word_connotation, word_gender):
    return "{}_{}_{}".format(word_type, word_connotation, word_gender)

  def invert_template_element(self, template_element):
    s = template_element.split("_")
    if len(s) != 3:
      raise ValueError("'{}' is not a valid template element".format(template_element))
    for s_i in s:
      if s_i == '':
        raise ValueError("'{}' is not a valid template element".format(template_element))
    return s[0], s[1], s[2]

  def __sanity_check(self, item, item_set):
    return item in item_set

  def __extract_template_elements(self, phrase):
    return re.findall('\{(.*?)\}',phrase)

  def __iterate_words(self, template_elements, template_element_index):
    is_last = template_element_index == (len(template_elements) - 1)
    template_element_key = template_elements[template_element_index]
    for word in self.__words[template_element_key]:
      if is_last:
        yield {template_element_key: word}
      else:
        for words in self.__iterate_words(template_elements, template_element_index+1):
          words[template_element_key] = word
          yield words
