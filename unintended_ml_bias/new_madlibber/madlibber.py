import re
import csv

class Madlibber(object):
  def __init__(self, path_helper, format_helper, word_helper):
    self.path_helper = path_helper
    self.format_helper = format_helper
    self.word_helper = word_helper
    self.__templates = []
    self.__template_elements = set([])
    self.__template_word_categories = set([])
    self.__toxicity = set(["toxic", "nontoxic"])

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
        template_element_word_categories = self.format_helper.decompose_template_element(t)
        self.__template_word_categories.update(template_element_word_categories)
        
      self.__templates.append((template, toxicity, phrase, template_elements))
      self.__template_elements.update(template_elements)
    f.close()

    print("Inferred word categories: {}".format(", ".join(self.__template_word_categories)))
    print("Done")

  def load_and_sanity_check_words(self):
    print("Loading word list...")
    f = open(self.path_helper.word_file, 'r')
    csv_f = csv.reader(f)
    header = next(csv_f)[:-1]  # Last column is the word itself

    for line in csv_f:
      word = line[-1]  # Last column is the word itself
      if word == '':
        continue

      unicode_word = word.decode('utf-8')
      for i, l in enumerate(line[:-1]):
        if l:
          word_category = self.format_helper.construct_word_category(header[i], l)
          self.word_helper.add_word(word_category, unicode_word)
    f.close()

    for t in self.__template_word_categories:
      if t not in self.word_helper.word_categories:
        raise ValueError("Template word_category '{}' is not represented in the word list".format(t))

    print("Word categories: {}".format(", ".join(self.word_helper.word_categories)))
    print("Done")

  def display_statistics(self):
    print("Template statistics:")
    print("Total number of templates: {}".format(len(self.__templates)))
    print("Total number of unique template elements to fill in: {}".format(len(self.__template_elements)))
    print("Word statistics:")
    template_element_word_counts = {}
    for i, te in enumerate(self.__template_elements):
      words = self.word_helper.get_template_element_words(te)
      template_element_word_counts[te] = len(words)
      print("Template element {}: {}, Total number of words: {}".format(i, te, template_element_word_counts[te]))
    total = 0
    for t in self.__templates:
      template_elements = t[-1]
      template_total = 1
      for te in template_elements:
        template_total *= template_element_word_counts[te]
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

  def __iterate_words(self, template_elements, template_element_index):
    is_last = template_element_index == (len(template_elements) - 1)
    template_element = template_elements[template_element_index]
    for word in self.word_helper.get_template_element_words(template_element):
      if is_last:
        yield {template_element: word}
      else:
        for words in self.__iterate_words(template_elements, template_element_index+1):
          words[template_element] = word
          yield words
