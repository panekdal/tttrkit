import numpy as np
from .file import Header

class TTTRReader:
    def __init__(self, path):
        self.path = path
        self._header = Header(path)  # parse ONCE
        self.offset = self._header.data_offset
        self.bits = self._header.bits_per_record
        self.dtype = np.uint32 if self.bits == 32 else np.uint64
        self._pos = self.offset  # current read position, advances on read/iter_chunks

    @property
    def header(self):
        return self._header

    def reset(self):
        """Reset the read position back to the start of the data (no re-parsing)."""
        self._pos = self.offset

    def read(self, count=None):
        with open(self.path, "rb") as f:
            f.seek(self._pos)
            data = np.fromfile(f, dtype=self.dtype, count=count)
            self._pos = f.tell()
        return data

    def iter_chunks(self, chunk_size=1000000, start=None):
        """Yield raw records in chunks of up to chunk_size.

        By default reading starts at the read position, which advances with
        each chunk. With ``start`` (a byte offset, e.g. ``self.offset`` for
        the start of the data) reading starts there and the read position
        is left untouched.
        """
        advance = start is None
        with open(self.path, "rb") as f:
            f.seek(self._pos if advance else start)
            while True:
                chunk = np.fromfile(f, dtype=self.dtype, count=chunk_size)
                if len(chunk) == 0:
                    break
                if advance:
                    self._pos = f.tell()
                yield chunk


