"""
Regression tests for upload ingestion of unreadable PDFs.

pypdf raises `PdfStreamError("Stream has ended unexpectedly")` on truncated or
corrupt PDFs — previously that surfaced as an opaque 500. `process_document`
must (a) salvage text from truncated files instead of crashing, and
(b) raise `UnreadablePdfError` for things that are not PDFs at all.
"""

import zlib

import pytest
from pypdf import PdfReader

from rag_engine import UnreadablePdfError, process_document


def _build_pdf(truncate: bool = False) -> bytes:
    """Builds a minimal but valid single-page PDF containing real text."""
    content = b"BT /F1 12 Tf 72 720 Td (Hello world, recovery test) Tj ET"
    objs = {
        1: b"<</Type/Catalog/Pages 2 0 R>>",
        2: b"<</Type/Pages/Kids[5 0 R]/Count 1>>",
        3: b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        4: zlib.compress(content),  # content stream
        5: (
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Resources<</Font<</F1 3 0 R>>>>/Contents 4 0 R>>"
        ),
    }

    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i in range(1, 6):
        offsets.append(len(buf))
        body = objs[i]
        if i == 4:
            buf += f"{i} 0 obj\n<< /Length {len(body)} /Filter /FlateDecode >>\nstream\n".encode()
            buf += body + b"\nendstream\nendobj\n"
        else:
            buf += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = b"xref\n0 6\n0000000000 65535 f \n"
    xref += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out = (
        bytes(buf)
        + xref
        + b"trailer\n<</Size 6/Root 1 0 R>>\nstartxref\n"
        + str(len(buf)).encode()
        + b"\n%%EOF\n"
    )
    if truncate:
        end = out.find(b"endstream")
        assert end > 0
        return out[: end + 40]
    return out


@pytest.fixture(autouse=True)
def _clean_tmp_files(tmp_path):
    yield
    for name in ("ok.pdf", "trunc.pdf", "bad.pdf"):
        p = tmp_path / name
        if p.exists():
            p.unlink()


class TestProcessDocument:

    def test_valid_pdf_parses_normally(self, tmp_path):
        p = tmp_path / "ok.pdf"
        p.write_bytes(_build_pdf())
        docs = process_document(str(p))
        assert len(docs) == 1
        assert "Hello world, recovery test" in docs[0].page_content
        assert docs[0].metadata["source"] == "ok.pdf"

    def test_truncated_pdf_is_recovered(self, tmp_path):
        p = tmp_path / "trunc.pdf"
        p.write_bytes(_build_pdf(truncate=True))
        # pre-condition: pypdf really does reject it
        with pytest.raises(Exception):
            PdfReader(str(p)).pages

        docs = process_document(str(p))
        assert len(docs) == 1
        assert "Hello world, recovery test" in docs[0].page_content

    def test_non_pdf_raises_unreadable_pdf_error(self, tmp_path):
        p = tmp_path / "bad.pdf"
        p.write_bytes(b"# this is not a pdf\nplain text")
        with pytest.raises(UnreadablePdfError):
            process_document(str(p))