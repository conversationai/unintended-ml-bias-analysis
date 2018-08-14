class TemplateWords(object):
  def __init__(self, template_element):
    self.__template_element = template_element
    self.__current_level_words = set([])
    self.__sub_template_words = {}

  def maybe_create_and_get_sub_template_words(self, sub_template_element):
    if sub_template_element not in self.__sub_template_words:
      self.__sub_template_words[sub_template_element] = TemplateWords(sub_template_element)
    return self.__sub_template_words[sub_template_element]

  def iterate_words(self):
    for w in self.__current_level_words:
      yield w
    for s in self.__sub_template_words.values():
      for w in s.iterate_words():
        yield w

  def add_word(self, word):
    self.__current_level_words.add(word)

  def template_element(self):
    return self.__template_element

  def sub_template_words(self):
    return self.__sub_template_words.keys()

  def current_level_words(self):
    return self.__current_level_words
