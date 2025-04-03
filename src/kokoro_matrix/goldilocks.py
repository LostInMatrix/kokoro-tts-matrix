import re
import logging
from typing import List, Tuple, Optional

class GoldilocksChunker:
    """
    Smart text chunker that creates TTS-optimized text segments by targeting the
    recommended token range while preserving natural sentence boundaries.
    """

    def __init__(
        self,
        min_tokens: int = 100,
        max_tokens: int = 200,
        too_short_threshold: int = 20,
        too_long_threshold: int = 400,
        sentence_splitter: str = r'(?<=[.!?]) +'
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.too_short_threshold = too_short_threshold
        self.too_long_threshold = too_long_threshold
        self.sentence_splitter = sentence_splitter
        self.logger = logging.getLogger(__name__)

    def estimate_tokens(self, text: str) -> int:
        """
        Roughly estimate number of tokens in the given text.
        A simple approximation based on word count and punctuation.
        """
        words = re.findall(r'\w+', text)
        punctuation = re.findall(r'[,.!?;:]', text)
        return len(words) + len(punctuation)

    def split_long_sentence(self, sentence: str) -> List[str]:
        """
        Intelligently split an overly long sentence at natural pause points.
        """
        if self.estimate_tokens(sentence) <= self.max_tokens:
            return [sentence]

        potential_splits = []
        for pattern in [r'(?<=, )', r'(?<=; )', r'(?<=: )', r' - ', r' -- ']:
            potential_splits.extend([(m.start(), pattern) for m in re.finditer(pattern, sentence)])

        if not potential_splits:
            self.logger.warning(f"Found overly long sentence with no good split points: {sentence[:100]}...")
            return [sentence]

        potential_splits.sort(key=lambda x: x[0])

        result = []
        last_pos = 0
        current_segment = ""

        for pos, pattern in potential_splits:
            candidate = sentence[last_pos:pos + len(pattern) - 1].strip()
            if self.estimate_tokens(current_segment + candidate) > self.max_tokens:
                if current_segment:
                    result.append(current_segment.strip())
                current_segment = candidate
            else:
                current_segment += candidate
            last_pos = pos + len(pattern) - 1

        if last_pos < len(sentence):
            final_segment = sentence[last_pos:].strip()
            if self.estimate_tokens(current_segment + final_segment) > self.max_tokens and current_segment:
                result.append(current_segment.strip())
                result.append(final_segment)
            else:
                result.append((current_segment + final_segment).strip())
        elif current_segment:
            result.append(current_segment.strip())

        return result

    def chunk_text(self, text: str) -> List[str]:
        """
        Break text into chunks optimized for TTS performance.
        Targets the goldilocks range while preserving sentence structure.
        """
        if not text or not text.strip():
            return []

        self.logger.debug(f"Chunking text with goldilocks range: {self.min_tokens}-{self.max_tokens} tokens")

        text_with_preserved_formatting = text.replace("\n\n", "<<DOUBLE_NEWLINE>>")
        text_for_processing = text_with_preserved_formatting.replace("\n", " ")
        raw_sentences = re.split(self.sentence_splitter, text_for_processing)
        sentences = []

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            token_estimate = self.estimate_tokens(sentence)
            self.logger.debug(f"Text has estimated {token_estimate} tokens before chunking")
            if token_estimate > self.too_long_threshold:
                self.logger.debug(f"Breaking long sentence ({token_estimate} tokens): {sentence[:50]}...")
                sentences.extend(self.split_long_sentence(sentence))
            else:
                sentences.append(sentence)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                if current_tokens < self.too_short_threshold:
                    current_chunk.append(sentence)
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

                if self.min_tokens <= current_tokens <= self.max_tokens:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        self.logger.debug(f"Created {len(chunks)} chunks from text with {len(sentences)} sentences")

        for i, chunk in enumerate(chunks):
            chunk_tokens = self.estimate_tokens(chunk)
            self.logger.debug(f"Chunk {i+1}: {chunk_tokens} tokens")

        return [chunk.replace("<<DOUBLE_NEWLINE>>", "\n\n") for chunk in chunks]