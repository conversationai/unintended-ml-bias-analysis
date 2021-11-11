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

import re


class FormatHelper():

  @classmethod
  def decompose_template_element(cls, template_element):
    s = template_element.split("_")
    for s_i in s:
      if not s_i or len(s_i.split("|")) != 2:
        raise ValueError(
            "'{}' is not a valid template element".format(template_element))
    return s

  @classmethod
  def extract_template_elements(cls, phrase):
    return re.findall(r"\{(.*?)\}", phrase)

  @classmethod
  def construct_word_category(cls, column_name, column_value):
    if not column_name:
      raise ValueError("'{}' is not a valid column name".format(column_name))
    if not column_value:
      raise ValueError("'{}' is not a valid column value".format(column_value))
    return "{}|{}".format(column_name, column_value)
