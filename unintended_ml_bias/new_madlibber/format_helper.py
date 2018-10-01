import re

class FormatHelper():
  @classmethod
  def decompose_template_element(cls, template_element):
    s = template_element.split("_")
    for s_i in s:
      if s_i == '' or len(s_i.split("|")) != 2:
        raise ValueError("'{}' is not a valid template element".format(template_element))
    return s

  @classmethod
  def extract_template_elements(cls, phrase):
    return re.findall('\{(.*?)\}',phrase)

  @classmethod
  def construct_word_category(cls, column_name, column_value):
    if not column_name:
      raise ValueError("'{}' is not a valid column name".format(column_name))
    if not column_value:
      raise ValueError("'{}' is not a valid column value".format(column_value))
    return "{}|{}".format(column_name, column_value)
