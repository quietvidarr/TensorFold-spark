"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.7     # a chain ends before a draft the MTP head gives less than this (one stream or many)
CONTEXT = 8192       # prompt plus reply tokens the caches hold

import os  # noqa: E402

# copy drafting: an exact repeat of the context (8+ matching tokens) is verified instead of the MTP chain when the MTP
# head's first draft agrees with the copy's first token (TF_FLASH_COPY=0: MTP only)
COPY = os.environ.get("TF_FLASH_COPY", "1") != "0"
