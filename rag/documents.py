"""Step 1 of RAG: turn files into chunks.

load_documents()  reads .md / .txt / .pdf files into plain text.
chunk_document()  cuts one document into overlapping chunks, remembering
                  which section heading each chunk came from.
"""
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    id: int          # position in the index (row number in the vectors matrix)
    source: str      # file name, used for citations
    section: str     # nearest heading above the chunk, e.g. "Sick leave"
    text: str        # the actual words

    def to_dict(self) -> dict:
        return asdict(self)

    def embedding_text(self) -> str:
        # "Contextual chunk header": we embed the file and section name along
        # with the text. A chunk that only says "You get 10 days." becomes
        # findable for "sick leave" because its header says "Sick leave".
        return f"{self.source} > {self.section}\n{self.text}"


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        # extract_text() can return None for scanned (image-only) pages.
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_documents(folder: Path) -> list[tuple[str, str]]:
    """Return [(file_name, text), ...] for every supported file in the folder."""
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = read_file(path).strip()
            if text:
                docs.append((path.name, text))
    return docs


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into paragraphs, tagging each with the heading it sits under.

    Paragraphs (separated by a blank line) are natural units of meaning, so we
    never cut inside one unless it is longer than a whole chunk.
    """
    section = "Introduction"
    blocks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        heading = re.match(r"^#{1,6}\s+(.*)", para)
        if heading:
            first_line, _, rest = para.partition("\n")
            section = re.sub(r"^#{1,6}\s+", "", first_line).strip()
            para = rest.strip()
            if not para:
                continue
        blocks.append((section, para))
    return blocks


def hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort for a giant paragraph: cut at sentence ends, or at spaces
    if a single sentence is itself too long."""
    units = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        units.extend(sentence.split() if len(sentence) > max_chars else [sentence])
    pieces, current = [], ""
    for u in units:
        if current and len(current) + len(u) + 1 > max_chars:
            pieces.append(current)
            current = u
        else:
            current = f"{current} {u}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(source: str, text: str, max_chars: int, overlap_chars: int,
                   start_id: int = 0) -> list[Chunk]:
    """Greedily pack paragraphs into chunks of at most max_chars.

    A new chunk starts when the section changes (so a chunk never mixes two
    topics) or when the next paragraph would not fit. The new chunk begins with
    the last `overlap_chars` of the previous one, so a fact that straddles the
    boundary is still complete in at least one chunk.
    """
    chunks: list[Chunk] = []
    current, current_section = "", None

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(Chunk(start_id + len(chunks), source, current_section, current.strip()))

    for section, para in split_into_blocks(text):
        for piece in hard_split(para, max_chars) if len(para) > max_chars else [para]:
            if section != current_section:
                flush()
                current, current_section = piece, section
            elif len(current) + len(piece) + 2 > max_chars:
                flush()
                tail = current[-overlap_chars:] if overlap_chars else ""
                # Start the overlap at a word boundary, not mid-word.
                tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                # Only add the overlap if the chunk still fits within max_chars.
                fits = tail and len(tail) + len(piece) + 2 <= max_chars
                current = f"{tail}\n\n{piece}" if fits else piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    flush()
    return chunks
