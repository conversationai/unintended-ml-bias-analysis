# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



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
