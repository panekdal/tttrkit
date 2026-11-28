# ptuio/file.py
import struct
from datetime import datetime, timedelta
from enum import IntEnum
from typing import BinaryIO
import numpy as np

from .decoder import T3_BITS_PER_RECORD, T3_WRAPAROUND, TTTRType

class PqFileType(IntEnum):
    PTU = 0x005254545051  # 'PQTTTR\0\0' as int
    PHU = 0x005453494851  # 'PQHISTO\0'

class TagTypes:
    tyEmpty8 = 0xFFFF0008
    tyBool8 = 0x00000008
    tyInt8 = 0x10000008
    tyBitSet64 = 0x11000008
    tyColor8 = 0x12000008
    tyFloat8 = 0x20000008
    tyTDateTime = 0x21000008
    tyFloat8Array = 0x2001FFFF
    tyAnsiString = 0x4001FFFF
    tyWideString = 0x4002FFFF
    tyBinaryBlob = 0xFFFFFFFF


def read_tag_value(file, tag_type):
    match tag_type:
        case TagTypes.tyEmpty8:
            file.read(8)  # skip 8 bytes
            return "<Empty>"

        case TagTypes.tyBool8:
            value = struct.unpack("q", file.read(8))[0]
            return bool(value)

        case TagTypes.tyInt8 | TagTypes.tyBitSet64 | TagTypes.tyColor8:
            return struct.unpack("q", file.read(8))[0]

        case TagTypes.tyFloat8:
            return struct.unpack("d", file.read(8))[0]

        case TagTypes.tyTDateTime:
            raw = struct.unpack("d", file.read(8))[0]
            date = datetime(1899, 12, 30) + timedelta(days=raw)
            return date

        case TagTypes.tyFloat8Array:
            length = struct.unpack("q", file.read(8))[0]
            file.seek(length, 1)  # skip the array data
            return f"<Float array with {length // 8} Entries>"

        case TagTypes.tyAnsiString | TagTypes.tyWideString:
            length = struct.unpack("q", file.read(8))[0]
            string_data = file.read(length)
            clean_string = string_data.replace(b"\x00", b"").decode(
                "utf-8", errors="ignore"
            )
            return clean_string

        case TagTypes.tyBinaryBlob:
            length = struct.unpack("q", file.read(8))[0]
            file.seek(length, 1)  # skip binary blob
            return f"<Binary Blob with {length} Bytes>"

        case _:
            raise ValueError(f"Unknown tag type: {hex(tag_type)}")


class Header:
    """Tags of a PicoQuant file header.

    Values the reconstruction depends on are read from the tags, never
    assumed: a missing tag raises when the value is used.
    """

    def __init__(self, path: str):
        self.path = path
        self.tags = {}
        self.file_type = None
        self.version = None
        self.record_type = None
        self.bits_per_record = None
        self.data_offset = 0

        with open(path, "rb") as f:
            self._read_header(f)

    def _read_header(self, f: BinaryIO):
        magic = f.read(8)
        if magic.startswith(b"PQTTTR"):
            self.file_type = PqFileType.PTU
        elif magic.startswith(b"PQHISTO"):
            self.file_type = PqFileType.PHU
        else:
            raise ValueError("Unsupported file type")

        self.version = f.read(8).rstrip(b"\0").decode("ascii", errors="ignore")

        while True:
            block = f.read(40)
            if len(block) < 40:
                raise ValueError("Incomplete tag block")
            tag_id_raw, index, typecode = struct.unpack("<32siI", block)
            tag_id = tag_id_raw.strip(b"\0").decode(errors="ignore")
            if tag_id == "Header_End":
                self.data_offset = f.tell()
                break
            if index >= 0:
                tag_key = f"{tag_id}[{index}]"
            else:
                tag_key = tag_id
            tag_value = read_tag_value(f, typecode)
            self.tags[tag_key] = tag_value

        self.record_type = self.tags.get("TTResultFormat_TTTRRecType")
        self.bits_per_record = self.tags.get("TTResultFormat_BitsPerRecord")

    def get(self, tag: str, default=None):
        return self.tags.get(tag, default)

    def require(self, tag: str):
        """The value of ``tag``; ValueError if the header lacks it."""
        if tag not in self.tags:
            raise ValueError(f"{self.path}: the header has no {tag} tag")
        return self.tags[tag]

    @property
    def sync_rate(self) -> float:
        """Laser sync (repetition) rate in Hz (TTResult_SyncRate)."""
        return float(self.require("TTResult_SyncRate"))

    @property
    def tcspc_resolution(self) -> float:
        """Seconds per dtime channel (MeasDesc_Resolution)."""
        return float(self.require("MeasDesc_Resolution"))

    def check_t3(self) -> None:
        """Raise unless the records are a T3 format that decode_t3 reads."""
        record_type = self.require("TTResultFormat_TTTRRecType")
        if record_type not in set(TTTRType):
            raise ValueError(
                f"{self.path}: unsupported record type {record_type:#010x}; "
                f"supported: "
                + ", ".join(f"{t.name} ({t.value:#010x})" for t in TTTRType)
            )
        bits = self.require("TTResultFormat_BitsPerRecord")
        if bits != T3_BITS_PER_RECORD:
            raise ValueError(
                f"{self.path}: {bits} bits per record, T3 records have "
                f"{T3_BITS_PER_RECORD}"
            )
        # Optional (Zeiss and Leica files lack it); the format fixes it
        wrap = self.tags.get("TTResultFormat_WrapAround", T3_WRAPAROUND)
        if wrap != T3_WRAPAROUND:
            raise ValueError(
                f"{self.path}: TTResultFormat_WrapAround={wrap}, but T3 "
                f"records wrap every {T3_WRAPAROUND} syncs"
            )

    def read_records(self, count=None):
        self.check_t3()
        with open(self.path, "rb") as f:
            f.seek(self.data_offset)
            return np.fromfile(f, dtype=np.uint32, count=count)

