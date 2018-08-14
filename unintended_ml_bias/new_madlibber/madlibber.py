import re
import csv

class Madlibber(object):
  def __init__(self, path_helper, format_helper, template_words_helper):
    self.path_helper = path_helper
    self.format_helper = format_helper
    self.tw_helper = template_words_helper
    self.__templates = []
    self.__template_elements = set([])
    self.__toxicity = set(["toxic", "nontoxic"])
    # word_types are things like identity, adjective, verb, occupation, etc
    self.__template_word_types = set([])
    # word connotations are like toxic, nontoxic, etc
    self.__template_word_connotations = set([])
    # word genders are masculine, feminine, etc for languages where this concept exists, i.e. in French blanc (masculine white) vs blanche (feminine white)
    self.__template_word_genders = set([])

  def load_sanity_check_templates_and_infer_word_categories(self):
    print("Loading templates, sanity checking and inferring word categories...")
    f = open(self.path_helper.sentence_template_file, 'r')
    csv_f = csv.DictReader(f)

    for line in csv_f:
      template = line['template']
      toxicity = line['toxicity']
      phrase = line['phrase']
      template_elements = self.format_helper.extract_template_elements(phrase)

      if phrase == '':
        continue

      if toxicity not in self.__toxicity:
        raise ValueError("'{}' is not a valid toxicity".format(toxicity))

      if len(template_elements) == 0:
        raise ValueError("Template '{}' does not require fill-in: '{}'".format(template, phrase))

      for t in template_elements:
        word_type, word_connotation, word_gender = self.format_helper.decompose_template_element(t)
        d_word_type = self.format_helper.deconstruct_word_type_hierarchy(word_type)
        for wt in self.__get_word_type_hierarchy(d_word_type):
          self.__template_word_types.add(wt)
        self.__template_word_connotations.add(word_connotation)
        self.__template_word_genders.add(word_gender)
        self.__template_elements.add(t)
        
      self.__templates.append((template, toxicity, phrase, template_elements))
    f.close()

    print("Inferred word types: {}".format(", ".join(self.__template_word_types)))
    print("Inferred word connotations: {}".format(", ".join(self.__template_word_connotations)))
    print("Inferred word genders: {}".format(", ".join(self.__template_word_genders)))
    print("Done")

  def load_and_sanity_check_words(self):
    print("Loading word list...")
    f = open(self.path_helper.word_file, 'r')
    csv_f = csv.DictReader(f)

    seen_template_elements = set([])
    for line in csv_f:
      word_type = line['type']
      word_conn = line['connotation']
      word_gender = line['gender']
      word = line['word']

      if word == '':
        continue

      d_word_type = self.format_helper.deconstruct_word_type_hierarchy(word_type)
      if not any(wt in self.__template_word_types for wt in self.__get_word_type_hierarchy(d_word_type)):
        raise ValueError("'{}' is not a valid word type".format(word_type))

      if word_conn not in self.__template_word_connotations:
        raise ValueError("'{}' is not a valid word connotation".format(word_conn))

      if word_gender not in self.__template_word_genders:
        raise ValueError("'{}' is not a valid word gender".format(word_gender))

      corresponds_to_template_element = False
      for wt in self.__get_word_type_hierarchy(d_word_type):
        template_element = self.format_helper.compose_template_element(wt, word_conn, word_gender)
        if template_element in self.__template_elements:
          seen_template_elements.add(template_element)
          corresponds_to_template_element = True
      if not corresponds_to_template_element:
        template_element = self.format_helper.compose_template_element(word_type, word_conn, word_gender)
        raise ValueError("'{}' is not a valid template element".format(template_element))

      unicode_word = word.decode('utf-8')
      template_element = self.format_helper.compose_template_element(word_type, word_conn, word_gender)
      template_words = self.tw_helper.maybe_create_and_get_template_words_for_element(template_element)
      template_words.add_word(unicode_word)
    f.close()

    for t in self.__template_elements:
      if t not in seen_template_elements:
        raise ValueError("Template element '{}' is not represented in the word list".format(t))
    print("Done")

  def display_statistics(self):
    print("Template statistics:")
    print("Total number of templates: {}".format(len(self.__templates)))
    print("Total number of unique template elements to fill in: {}".format(len(self.__template_elements)))
    print("Word statistics:")
    counts = self.tw_helper.count_template_words()
    for i, te in enumerate(self.__template_elements):
      print("Template element {}: {}, Total number of words: {}".format(i, te, counts[te]))
    total = 0
    for t in self.__templates:
      template_elements = t[-1]
      template_total = 1
      for te in template_elements:
        template_total *= counts[te]
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

  def __get_word_type_hierarchy(self, deconstructed_word_type):
    for i in range(0, len(deconstructed_word_type)):
      yield self.format_helper.construct_word_type_hierarchy(deconstructed_word_type[:i+1])

  def __iterate_words(self, template_elements, template_element_index):
    is_last = template_element_index == (len(template_elements) - 1)
    template_element = template_elements[template_element_index]
    word_type, word_conn, word_gender = self.format_helper.decompose_template_element(template_element)
    template_words = self.tw_helper.maybe_create_and_get_template_words_for_element(template_element)
    for word in template_words.iterate_words():
      if is_last:
        yield {template_element: word}
      else:
        for words in self.__iterate_words(template_elements, template_element_index+1):
          words[template_element] = word
          yield words
