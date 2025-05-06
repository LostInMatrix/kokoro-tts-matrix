import os
import logging
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from typing import Dict, Union
import base64
from datetime import datetime
import re
from .goldilocks import GoldilocksChunker
from .pronunciation_dictionary import PronunciationDictionary

_goldilocks_chunker = None

logging.basicConfig(filename='audio_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_epub(epub_file: str) -> str:
    """
    Extract full text from an EPUB file.

    :param epub_file: Path to the EPUB file.
    :return: The extracted text.
    """
    book = epub.read_epub(epub_file)
    full_text = ""
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            full_text += soup.get_text()
    return full_text

import re
import logging

_pronunciation_dict = None

def get_pronunciation_dictionary():
    global _pronunciation_dict
    if _pronunciation_dict is None:
        _pronunciation_dict = PronunciationDictionary()
    return _pronunciation_dict

def apply_pronunciation_dictionary(text):
    return get_pronunciation_dictionary().apply_pronunciations(text)

def convert_all_caps(text: str) -> str:
    """Convert ALL CAPS words to Title Case for more natural TTS reading."""

    lines = text.split('\n')
    result = []

    for line in lines:
        words = line.split()
        converted_words = []

        for word in words:
            if word.isupper() and len(word) > 1:
                common_acronyms = ['USA', 'UK', 'EU', 'UN', 'FBI', 'CIA', 'NASA', 'NATO',
                                   'CEO', 'CFO', 'CTO', 'PhD', 'AI', 'ML', 'TTS', 'PDF',
                                   'HTML', 'CSS', 'JS', 'API']

                if word in common_acronyms:
                    converted_words.append(word)
                else:
                    converted_words.append(word.title())
            else:
                converted_words.append(word)

        result.append(' '.join(converted_words))

    return '\n'.join(result)

def chunk_text(text: str, chunk_size: int = 2000, use_goldilocks: bool = True) -> list:
    """
    Break text into chunks with intelligent handling based on TTS optimization.

    Args:
        text: The text to chunk
        chunk_size: Legacy parameter for character-based chunking
        use_goldilocks: Whether to use token-based chunking (recommended)

    Returns:
        List of text chunks optimized for TTS
    """
    global _goldilocks_chunker

    logging.debug(f"chunk_text called with text (first 100 chars): {text[:100]}...")
    logging.debug(f"chunk_size: {chunk_size}, use_goldilocks: {use_goldilocks}")

    if not text or not text.strip():
        logging.debug("chunk_text: Empty text, returning []")
        return []

    text = apply_pronunciation_dictionary(text)

    if use_goldilocks:
        if _goldilocks_chunker is None:
            _goldilocks_chunker = GoldilocksChunker()
            logging.debug(f"Initialized GoldilocksChunker with token range: {_goldilocks_chunker.min_tokens}-{_goldilocks_chunker.max_tokens}")

        chunks = _goldilocks_chunker.chunk_text(text)
        logging.debug(f"GoldilocksChunker created {len(chunks)} chunks")
        return chunks

    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    sentences = [s.strip() for s in sentences]  # Strip whitespace
    logging.debug(f"chunk_text: Sentences after split and strip: {sentences}")

    if not sentences:
        logging.debug("chunk_text: No sentences, returning []")
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    for i, sentence in enumerate(sentences, 1):
        logging.debug(f"--- Processing sentence {i}/{len(sentences)} ---")
        logging.debug(f"  Sentence (full): {sentence}")
        sentence_size = len(sentence)
        logging.debug(f"  Sentence size: {sentence_size}")
        logging.debug(f"  Current chunk: {current_chunk}")
        logging.debug(f"  Current size: {current_size}")

        if current_size + sentence_size > chunk_size:
            logging.debug("  Chunk size EXCEEDED")
            if current_chunk:
                logging.debug(f"    Appending chunk: {' '.join(current_chunk)}")
                chunks.append(' '.join(current_chunk))
            else:
                logging.debug("    Current chunk is empty (shouldn't happen)")
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size
        logging.debug(f"  New current chunk: {current_chunk}")
        logging.debug(f"  New current size: {current_size}")

    if current_chunk:
        logging.debug("Appending final chunk")
        chunks.append(' '.join(current_chunk))

    logging.debug(f"chunk_text: Returning chunks: {chunks}")
    return chunks

def extract_chapters_from_epub(epub_file: str) -> list:
    """
    Extract chapters from an EPUB file using its table of contents.

    :param epub_file: Path to the EPUB file.
    :return: A list of chapter dictionaries with keys 'title', 'content', and 'order'.
    """
    if not os.path.exists(epub_file):
        raise FileNotFoundError(f"EPUB file not found: {epub_file}")

    book = epub.read_epub(epub_file)
    chapters = []

    def process_toc_items(items, depth=0):
        processed = []
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                section_title, section_items = item
                logging.debug(f"{'  ' * depth}Processing section: {section_title}")
                processed.extend(process_toc_items(section_items, depth + 1))
            elif hasattr(item, 'title'):
                logging.debug(f"{'  ' * depth}Processing link: {item.title} -> {item.href}")
                if item.title.lower() in ['copy', 'copyright', 'title page', 'cover'] or item.title.lower().startswith('by'):
                    continue
                href_parts = item.href.split('#')
                file_name = href_parts[0]
                fragment_id = href_parts[1] if len(href_parts) > 1 else None
                doc = next((doc for doc in book.get_items_of_type(ITEM_DOCUMENT)
                            if doc.file_name.endswith(file_name)), None)
                if doc:
                    content = doc.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, "html.parser")
                    if not fragment_id:
                        text_content = soup.get_text().strip()
                    else:
                        elem = soup.find(id=fragment_id)
                        text_content = elem.get_text().strip() if elem else ""
                    if text_content:
                        chapters.append({
                            'title': item.title,
                            'content': text_content,
                            'order': len(chapters) + 1
                        })
                        processed.append(item)
                        logging.debug(f"{'  ' * depth}Added chapter: {item.title}")
        return processed

    process_toc_items(book.toc)

    if not chapters:
        logging.warning("No chapters found in TOC. Consider processing the full text instead.")

    return chapters

def extract_epub_metadata(epub_file: str) -> Dict[str, Union[str, int, float]]:
    """
    Extract metadata from an EPUB file.
    Returns basic metadata plus calculated fields like word count and duration.
    """
    try:
        book = epub.read_epub(epub_file)

        metadata = {
            'title': book.title,
            'language': book.language,
            'author': next((value for name, value in book.get_metadata('DC', 'creator')), None),
            'publisher': next((value for name, value in book.get_metadata('DC', 'publisher')), None),
            'publication_date': next((value for name, value in book.get_metadata('DC', 'date')), None)
        }

        total_words = sum(
            len(re.sub('<[^<]+?>', '', item.get_content().decode('utf-8')).split())
            for item in book.get_items_of_type(ITEM_DOCUMENT)
        )
        metadata['word_count'] = total_words

        if total_words > 0:
            metadata['estimated_duration'] = round(total_words / 150, 2)

        logging.debug(f"Successfully extracted metadata: {metadata}")
        return metadata

    except Exception as e:
        logging.error(f"Error extracting EPUB metadata: {e}")
        raise

def convert_initial_caps(text: str) -> str:
    """Convert leading all-caps words that are likely names/chapter starts."""

    lines = text.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue

        if stripped.startswith(('CHAPTER', 'BOOK', 'PART')):
            leading_space = len(line) - len(stripped)
            result.append(' ' * leading_space + stripped.title())
            continue

        words = stripped.split()
        if not words:
            result.append(line)
            continue

        first_word = words[0]
        if (first_word.isupper() and len(first_word) > 1 and
            len(words) > 1 and
            not words[1].isupper()):

            leading_space = len(line) - len(stripped)
            words[0] = first_word.title()
            result.append(' ' * leading_space + ' '.join(words))
        else:
            result.append(line)

    return '\n'.join(result)