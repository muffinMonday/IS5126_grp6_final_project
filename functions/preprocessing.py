import re
import string
import unicodedata

# irregular forms first, then the general suffixes; "'s" is dropped (possessive or "is")
CONTRACTIONS = [
  (r"\bwon't\b", 'will not'),
  (r"\bcan't\b", 'can not'),
  (r"\bshan't\b", 'shall not'),
  (r"\bain't\b", 'is not'),
  (r"n't\b", ' not'),
  (r"'re\b", ' are'),
  (r"'ve\b", ' have'),
  (r"'ll\b", ' will'),
  (r"'d\b", ' would'),
  (r"'m\b", ' am'),
  (r"'s\b", ''),
]


def cleanup_text(text):
  # 1. Unicode normalization (converts quirky characters to standard form)
  text = unicodedata.normalize('NFKC', text)

  # 2. Remove HTML tags
  text = re.sub(r'<.*?>', '', text)

  # 3. Remove URLs
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

  # 4. Expand contractions into words GloVe knows ("don't" -> "do not")
  text = text.replace('’', "'").replace('‘', "'")
  for pattern, replacement in CONTRACTIONS:
    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

  # 5. Keep ! and ? as separate tokens, they carry tone
  text = re.sub(r'([!?])', r' \1 ', text)

  # 6. Replace the remaining punctuation with a space so words don't glue together
  #    ("self-entitlement" -> "self entitlement"); leftover apostrophes are just removed
  translator = {ord(c): ' ' for c in string.punctuation if c not in "!?'"}
  translator[ord("'")] = None
  text = text.translate(translator)

  # 7. Clean up extra spaces left behind from removed tags/URLs/punctuation
  text = re.sub(r'\s+', ' ', text).strip()

  return text
