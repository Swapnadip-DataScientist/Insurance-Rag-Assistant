import re
from typing import List, Dict, Any

def split_into_paragraphs(text : str) -> List[str]:
    """
    Split cleaned text into paragraphs. Paragraph boundaries are identified using one or more blank lines.  
    Empty paragraphs are removed.
    """

    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)

    return [ paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def split_into_sentences(text : str) -> List[str]:

    """
    Split a paragraph into sentences. This is intentionally lightweight for the baseline implementation.
    It avoids introducing an additional NLP dependency during ingestion.
    """

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())

    return [sentence.strip() for sentence in sentences if sentence.strip()]

def create_overlap( previous_chunk_text : str, overlap_chars:int):

    """Extract overlap text from the end of the previous chunk.Tries to avoid cutting directly through a word.
    """

    if not previous_chunk_text or overlap_chars <=0 :

        return ""

    if len(previous_chunk_text) < overlap_chars :

        return previous_chunk_text

    overlap_text = previous_chunk_text[-overlap_chars:]

    first_space = overlap_text.find("")

    if first_space != -1 :
        overlap_text = overlap_text[first_space+1:]

    return overlap_text.strip()


def chunk_text( text:str, max_chunk_chars: int = 500, overlap_chars : int = 50) -> List[str]:

    """
    Chunk text using paragraph-first and sentence-fallback strategy.
    Strategy:
        1. Preserve paragraphs whenever possible.
        2. Combine paragraphs until max_chunk_chars is reached.
        3. If one paragraph itself is too large: - split it into sentences.
        4. Maintain character-based overlap between chunks.

    This is the baseline chunking implementation for the RAG pipeline.

    Parameters
    ----------
    text: Cleaned document/page text.
    max_chunk_chars: Maximum approximate chunk size in characters.
    overlap_chars: Number of characters copied from the previous chunk.
    Returns 
    -------
    List[str] 
        Chunk text strings.
    """

    if not text or not text.strip():
        return []

    if max_chunk_chars <= 0:
        raise ValueError ("Maximum Chunk Characters must be greater than 0")

    if overlap_chars <0:
        raise ValueError ("Overlap character cannot be 0")

    if overlap_chars >= max_chunk_chars:
        raise ValueError ("Overlap Characters must ne smaller than Max Chunk Characters")

    paragraphs = split_into_paragraphs(text)

    chunks : List[str] = []
    current_parts : List[str] = []
    current_length = 0

    def flush_current_chunk():

        nonlocal current_parts
        nonlocal current_length

        if not current_parts:
            return

        chunk = "\n\n".join(current_parts).strip()

        if chunk:
            chunks.append(chunk)

        current_parts = []
        current_length = 0

    for paragraph in paragraphs:

        """Normal paragraph"""

        if len(paragraph) <= max_chunk_chars:

            separator_length_= 2 if current_parts else 0
            projected_length = current_length + separator_length_ + len(paragraph)

            if projected_length <= max_chunk_chars:
                current_parts.append(paragraph)
                current_length = projected_length
            else:
                flush_current_chunk()
                # Add overlap from previous chunk
                if chunks and overlap_chars >0 :
                    overlap_text = create_overlap(chunks[-1], overlap_chars)
                    if overlap_text:
                        current_parts.append(overlap_text)
                        current_length = len(overlap_text)
                separator_length = 2 if current_parts else 0
                # If paragraph + overlap still does not fit,
                # store overlap separately only when necessary.
                if ((current_length + separator_length + len(paragraph)) > max_chunk_chars):
                    flush_current_chunk()
                current_parts.append(paragraph)
                current_length += ((2 if current_length else 0)+ len(paragraph))

        #************Oversize*************
        else:
            flush_current_chunk()
            sentences = split_into_sentences(paragraph)


            # Sentence splitter may fail for text containing no
            # punctuation. Fall back to character splitting.
            print("sentences type-> ", type(sentences))
            if len(sentences) <= 1 and len(paragraph) > max_chunk_chars:

                start = 0

                while start < len(paragraph):

                    end = start + max_chunk_chars

                    piece = paragraph[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    start = end - overlap_chars

                continue

            sentence_parts: List[str] = []
            sentence_length = 0

            for sentence in sentences:

                # Extremely large individual sentence fallback
                if len(sentence) > max_chunk_chars:

                    if sentence_parts:

                        chunk = " ".join(sentence_parts).strip()

                        if chunk:
                            chunks.append(chunk)

                        sentence_parts = []
                        sentence_length = 0

                    start = 0

                    while start < len(sentence):

                        end = start + max_chunk_chars

                        piece = sentence[start:end].strip()

                        if piece:
                            chunks.append(piece)

                        start = end - overlap_chars

                    continue

                separator_length = 1 if sentence_parts else 0

                projected_length = (
                    sentence_length
                    + separator_length
                    + len(sentence)
                )

                if projected_length <= max_chunk_chars:

                    sentence_parts.append(sentence)
                    sentence_length = projected_length

                else:

                    chunk = " ".join(sentence_parts).strip()

                    if chunk:
                        chunks.append(chunk)

                    overlap_text = create_overlap(
                        chunk,
                        overlap_chars
                    )

                    sentence_parts = []

                    if overlap_text:
                        sentence_parts.append(overlap_text)

                    sentence_parts.append(sentence)

                    sentence_length = len(
                        " ".join(sentence_parts)
                    )

            if sentence_parts:

                chunk = " ".join(sentence_parts).strip()

                if chunk:
                    chunks.append(chunk)

    flush_current_chunk()

    return chunks



def chunk_pages(pages : List[Dict [str, Any]], document_id : str, source_file : str, max_chunk_chars : int = 500,
                overlap_chars : int = 200) -> List [Dict[str, Any]]:

    """
    Chunk page-level document data.
    Expected page structure:
    {
        "page_number": 1,
        "text": "...",
        ...
    }
    Returns chunk dictionaries containing metadata required
    later for embeddings, vector database storage and citations.
    """

    all_chunks : List [Dict[str,Any]] = []

    document_chunk_index= 0

    for page in pages:

        page_number = page.get("page_number")

        text = page.get("text","")

        if not text or not text.strip():
            continue 

        page_chunks = chunk_text(text= text, max_chunk_chars = max_chunk_chars,overlap_chars=overlap_chars)

        for page_chunk_index, chunk in enumerate(page_chunks):

            chunk_id = (f"{document_id}_" f"p{page_number}_"f"c{page_chunk_index:04d}"
                        )

            chunk_record = {
                "chunk_id" : chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "page_number": page_number,
                "page_chunk_index" : page_chunk_index,
                "text": chunk,
                "character_count":len(chunk),
                "chunking_strategy" : ("paragraph_sentence_overlap"),
                "max_chunk_chars":max_chunk_chars,
                "overlap_chars": overlap_chars
            }

            all_chunks.append(chunk_record)

            document_chunk_index += 1
    return all_chunks