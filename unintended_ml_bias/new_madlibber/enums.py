def enum(**enums):
  values = set([value for value in enums.values()])
  enums['exists'] = values.__contains__
  return type('Enum', (), enums)

Toxicity = enum(TOXIC='toxic', NONTOXIC= 'non-toxic')
WordType = enum(NAME='name', ADJECTIVE='adjective', IDENTITY='identity', OCCUPATION='occupation', VERB='verb')
WordConnotation = enum(POSITIVE='positive', NEGATIVE='negative', NEUTRAL='neutral')
WordGender = enum(MALE='m', FEMALE='f', NEITHER='n')
