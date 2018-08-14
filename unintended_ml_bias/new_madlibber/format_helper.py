import re

class FormatHelper():
  @classmethod
  def compose_template_element(cls, word_type, word_connotation, word_gender):
    return "{}_{}_{}".format(word_type, word_connotation, word_gender)

  @classmethod
  def decompose_template_element(cls, template_element):
    s = template_element.split("_")
    if len(s) != 3:
      raise ValueError("'{}' is not a valid template element".format(template_element))
    for s_i in s:
      if s_i == '':
        raise ValueError("'{}' is not a valid template element".format(template_element))
    return s[0], s[1], s[2]

  @classmethod
  def construct_word_type_hierarchy(cls, split_word_type):
    return "|".join(split_word_type)

  @classmethod
  def deconstruct_word_type_hierarchy(cls, word_type):
    d = word_type.split("|")
    for d_i in d:
      if d_i == '':
        raise ValueError("'{}' is not a valid word category".format(word_type))
    return d

  @classmethod
  def extract_template_elements(cls, phrase):
    return re.findall('\{(.*?)\}',phrase)
