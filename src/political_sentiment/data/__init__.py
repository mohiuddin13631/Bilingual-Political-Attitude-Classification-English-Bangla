"""Data layer: schema, synthetic generation, real-dataset loaders, splitting."""

from .schema import RECORD_COLUMNS, SentimentRecord, records_to_frame
from .split import make_splits

__all__ = ["SentimentRecord", "RECORD_COLUMNS", "records_to_frame", "make_splits"]
