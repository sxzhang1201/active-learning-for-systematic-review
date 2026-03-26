"""
Fetch PubMed records via NCBI E-utilities (replaces legacy PmedConnect).

Requires NCBI-compliant use: set config.PUBMED_EMAIL (and optionally an API key via
environment variable NCBI_API_KEY for higher rate limits).
"""
import calendar
import os
import random
import re
import time
from http.client import RemoteDisconnected
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

try:
  from Bio import Entrez, Medline
except ModuleNotFoundError as exc:
  raise ModuleNotFoundError(
      "PubMed fetch needs Biopython (import name `Bio`). "
      "Install the *PyPI package* `biopython`, e.g.:\n"
      "  python -m pip install biopython\n"
      "Use the same interpreter you use to run fetch_raw.py (check with `which python`)."
  ) from exc

import config


def _txt(val: Any) -> Optional[str]:
  if val is None:
    return None
  if isinstance(val, list):
    if not val:
      return None
    return ' '.join(str(x) for x in val)
  return str(val).strip() or None


def _month_num(token: Optional[str]) -> Optional[str]:
  if not token:
    return None
  token = token.strip()
  mapping = {abbr: str(i).zfill(2) for i, abbr in enumerate(calendar.month_abbr) if abbr}
  if token in mapping:
    return mapping[token]
  mapping_full = {name.lower(): str(i).zfill(2) for i, name in enumerate(calendar.month_name) if name}
  if token.lower() in mapping_full:
    return mapping_full[token.lower()]
  if token.isdigit() and 1 <= int(token) <= 12:
    return token.zfill(2)
  return None


def _parse_dp(dp: Optional[str]) -> tuple:
  """Return (year, month, day) string components from MEDLINE DP field."""
  if not dp:
    return None, None, None
  dp = dp.strip()
  parts = dp.replace('.', '').split()
  if not parts:
    return None, None, None
  year = parts[0] if parts[0].isdigit() else None
  month = _month_num(parts[1]) if len(parts) > 1 else None
  day = parts[2].zfill(2) if len(parts) > 2 and parts[2].isdigit() else None
  return year, month, day


def _doi_from_record(record: Dict) -> Optional[str]:
  for key in ('LID', 'AID'):
    raw = record.get(key)
    items = raw if isinstance(raw, list) else ([raw] if raw else [])
    for item in items:
      s = str(item)
      if '[doi]' in s.lower():
        return s.split()[0].strip()
  return None


def _issn_from_record(record: Dict) -> Optional[str]:
  raw = record.get('IS')
  if not raw:
    return None
  s = _txt(raw)
  if not s:
    return None
  m = re.match(r'^(\d{4}-?\d{3}[\dX])', s.replace(' ', ''))
  return m.group(1) if m else None


def medline_record_to_article(record: Dict) -> Dict[str, Any]:
  """Shape expected by Database.db_inserts.Connector.insert_fetched_articles."""
  pmid = _txt(record.get('PMID'))
  title = _txt(record.get('TI')) or ''
  abstract = _txt(record.get('AB'))
  jt = _txt(record.get('JT'))
  ta = _txt(record.get('TA')) or ''
  journal_title = jt or ta or None
  journal_iso = ta or jt or ''
  journal_issn = _issn_from_record(record)
  dp = _txt(record.get('DP'))
  pub_year, pub_month, pub_day = _parse_dp(dp)

  return {
    'pmid': pmid,
    'title': title,
    'abstract': abstract,
    'doi': _doi_from_record(record),
    'journal_title': journal_title,
    'journal_iso': journal_iso if journal_iso else 'unknown',
    'journal_issn': journal_issn,
    'pub_year': pub_year,
    'pub_month': pub_month,
    'pub_day': pub_day,
    'date_pubmed_published': _txt(record.get('PHST')),
    'date_medline_published': _txt(record.get('DA')),
  }


def _empty_article(pmid: str) -> Dict[str, Any]:
  return {
    'pmid': pmid,
    'title': '',
    'abstract': None,
    'doi': None,
    'journal_title': None,
    'journal_iso': 'unknown',
    'journal_issn': None,
    'pub_year': None,
    'pub_month': None,
    'pub_day': None,
    'date_pubmed_published': None,
    'date_medline_published': None,
  }


def _configure_entrez(email: str) -> None:
  Entrez.email = email
  Entrez.tool = config.PUBMED_TOOL_NAME
  api_key = os.environ.get('NCBI_API_KEY')
  if api_key:
    Entrez.api_key = api_key


def _rate_sleep() -> None:
  # NCBI: ~3 req/s without API key, ~10/s with key; stay conservative after each efetch chunk.
  time.sleep(0.12 if os.environ.get('NCBI_API_KEY') else 0.35)


def _http_retryable(exc: BaseException) -> bool:
  if isinstance(exc, HTTPError):
    return exc.code in (429, 500, 502, 503, 504)
  return isinstance(exc, (RemoteDisconnected, URLError, OSError, TimeoutError))


def _efetch_medline(id_str: str):
  """Open Medline efetch handle with retries (NCBI often resets idle or busy connections)."""
  retries = max(1, config.PUBMED_EFETCH_RETRIES)
  last = None
  for attempt in range(retries):
    try:
      return Entrez.efetch(db='pubmed', id=id_str, rettype='medline', retmode='text')
    except (RemoteDisconnected, URLError, HTTPError, OSError) as exc:
      last = exc
      if not _http_retryable(exc):
        raise
      if attempt >= retries - 1:
        break
      wait = min(120.0, (2 ** attempt) + random.uniform(0.25, 2.0))
      print('Warning: PubMed efetch error ({!s}); sleep {:.1f}s ({}/{})'.format(
          exc, wait, attempt + 1, retries))
      time.sleep(wait)
  raise last


def fetch_pubmed_ids(pubmed_ids: List[str], email: str, chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
  """
  Return one article dict per input PMID, same order (matches qrel batch rows).
  Missing records get placeholder entries so lengths stay aligned with insert_docs.
  """
  _configure_entrez(email)
  if not pubmed_ids:
    return []

  if chunk_size is None:
    chunk_size = max(10, config.PUBMED_CHUNK_SIZE)

  normalized = [str(p).strip() for p in pubmed_ids]
  by_pmid: Dict[str, Dict[str, Any]] = {}

  for i in range(0, len(normalized), chunk_size):
    chunk = normalized[i:i + chunk_size]
    unique_ids = list(dict.fromkeys(chunk))
    id_str = ','.join(unique_ids)
    handle = _efetch_medline(id_str)
    try:
      for record in Medline.parse(handle):
        article = medline_record_to_article(record)
        p = article.get('pmid')
        if p:
          by_pmid[str(p)] = article
    finally:
      handle.close()
    _rate_sleep()

  return [by_pmid.get(p, _empty_article(p)).copy() for p in normalized]


class PubmedAPI:
  """Drop-in style replacement for PmedConnect.PubmedAPI used by fetch_raw."""

  def __init__(self, email: str):
    self.email = email

  def fetch(self, pubmed_ids: List[str]) -> List[Dict[str, Any]]:
    return fetch_pubmed_ids(pubmed_ids, self.email)
