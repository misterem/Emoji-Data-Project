# from pathlib import Path
# import csv
#
# def first_bad_line(path: str | Path, encoding: str = "utf-8"):
#     """Print the first line that fails to decode with `encoding`."""
#     path = Path(path)
#     with path.open("rb") as f:                      # read *binary* to avoid decoding up front
#         for lineno, raw in enumerate(f, 1):
#             try:
#                 raw.decode(encoding)               # try full-line UTF-8 decode
#             except UnicodeDecodeError as err:
#                 # Show where the failure starts and a few following bytes
#                 bad_slice = raw[err.start : err.start + 10]
#                 print(f"{path.name}  line {lineno}  byte {err.start}: {bad_slice!r}")
#                 return raw, lineno                 # hand the raw bytes back for deeper inspection
#     print("✓ no decoding problems found")
#
# def find_bad_rows(path, encoding="utf-8", delimiter=","):
#     """
#     Report lines that do not have the same number of columns as the header
#     (typical symptom of an unescaped newline or rogue quote).
#     """
#     path = Path(path)
#     with path.open("r", encoding=encoding, newline="") as f:
#         reader = csv.reader(f, delimiter=delimiter)
#         header = next(reader)              # first line defines the column count
#         expected_cols = len(header)
#
#         for lineno, row in enumerate(reader, start=2):
#             if len(row) != expected_cols:  # mismatch = malformed
#                 preview = ",".join(row)[:120]
#                 print(f"{path.name}: line {lineno} has {len(row)} cols "
#                       f"(expected {expected_cols}) → {preview!r}")
#
# find_bad_rows("./archive/backhand_index_pointing_right.csv", encoding="utf-8")
#
# # bad_raw, bad_lineno = first_bad_line("./negative-words.txt")  # or any file that crashes

import matplotlib.font_manager
available_fonts = sorted({f.name for f in matplotlib.font_manager.fontManager.ttflist})
print(available_fonts)
