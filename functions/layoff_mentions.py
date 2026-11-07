"""Layoff matching rules restored from the original Python 3.14 bytecode cache."""

import re


LAYOFF_TERMS = {
  'laid_off': ('"laid off"', r'\blaid[\s-]+(?:[a-z]+\s+){0,2}?off\b'),
  'layoff': ('layoff', r'\blay[\s-]?offs?\b'),
  'severance': ('severance', r'\bseverance\b'),
  'downsizing': ('downsizing', r'\bdownsiz(?:e|ed|es|ing)\b'),
  'redundancy': ('redundancy', r'\bredundanc(?:y|ies)\b|\bmade\s+redundant\b'),
}
_PATTERNS = {term: re.compile(pattern, re.IGNORECASE)
             for term, (_, pattern) in LAYOFF_TERMS.items()}


def layoff_terms(text, terms=None):
  """Return the matching term names, in the requested order."""
  if not isinstance(text, str):
    return []
  return [term for term in (terms or LAYOFF_TERMS) if _PATTERNS[term].search(text)]
