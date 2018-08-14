class TemplateWordsHelper(object):
  def __init__(self, format_helper, template_words_class):
    self.format_helper = format_helper
    self.template_words_class = template_words_class
    self.__root_level_template_elements = {}
    self.__all_template_elements = {}

  def maybe_create_and_get_template_words_for_element(self, template_element):
    if template_element in self.__all_template_elements:
      return self.__all_template_elements[template_element]
    word_type, word_conn, word_gender = self.format_helper.decompose_template_element(template_element)
    d_word_type = self.format_helper.deconstruct_word_type_hierarchy(word_type)
    root_template_element = self.format_helper.compose_template_element(d_word_type[0], word_conn, word_gender)
    if root_template_element not in self.__root_level_template_elements:
      self.__root_level_template_elements[root_template_element] = self.template_words_class(root_template_element)
      self.__all_template_elements[root_template_element] = self.__root_level_template_elements[root_template_element]
    words = self.__root_level_template_elements[root_template_element]
    for i in range(1, len(d_word_type)):
      word_type = self.format_helper.construct_word_type_hierarchy(d_word_type[:i+1])
      te = self.format_helper.compose_template_element(word_type, word_conn, word_gender)
      if te in self.__all_template_elements:
        words = self.__all_template_elements[te]
      else:
        words = words.maybe_create_and_get_sub_template_words(te)
        self.__all_template_elements[te] = words
    return words

  def count_template_words(self):
    memoized_counts = {}
    for template_element, words in self.__root_level_template_elements.iteritems():
      self.__count_template_words(template_element, words, memoized_counts)
    return memoized_counts

  def __count_template_words(self, template_element, words, memoized_counts):
    if template_element in memoized_counts:
      return memoized_counts[template_element]
    n = len(words.current_level_words())
    if words.sub_template_words():
      for st in words.sub_template_words():
        n += self.__count_template_words(st, words.maybe_create_and_get_sub_template_words(st), memoized_counts)
    memoized_counts[template_element] = n
    return n
