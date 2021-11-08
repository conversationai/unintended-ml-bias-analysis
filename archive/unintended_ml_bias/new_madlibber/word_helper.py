

class WordHelper(object):

  def __init__(self, format_helper):
    self.format_helper = format_helper
    self.word_category_words = {}

  def add_word(self, word_category, word):
    self.word_category_words.setdefault(word_category, set([]))
    self.word_category_words[word_category].add(word)

  def get_template_element_words(self, template_element):
    template_element_word_categories = (
        self.format_helper.decompose_template_element(template_element))
    words = self.word_category_words[template_element_word_categories[0]]
    for tewc in template_element_word_categories[1:]:
      words = words.intersection(self.word_category_words[tewc])
    return words

  @property
  def word_categories(self):
    return self.word_category_words.keys()
