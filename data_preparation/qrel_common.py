"""Shared qrel batching for fetch_raw and insert_docs (must stay in sync)."""
import config

QREL_BATCH_SIZE = 50000


def effective_fetch_limit():
  lim = config.FETCH_RAW_MAX_ARTICLES or None
  if lim is not None and lim <= 0:
    return None
  return lim


def next_batch_size(running_total, max_articles):
  if max_articles is None:
    return QREL_BATCH_SIZE
  return min(QREL_BATCH_SIZE, max(1, max_articles - running_total))


def over_limit(running_total, max_articles):
  return max_articles is not None and running_total >= max_articles
