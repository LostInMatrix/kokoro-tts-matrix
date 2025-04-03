"""
Module for processing audio generation and file saving.
"""

import os
import sys
import time
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from kokoro_matrix.pipeline import initialize_pipeline
from kokoro_matrix.text_processing import chunk_text, extract_chapters_from_epub
from .voice_segments import (
    VoiceSegment,
    parse_voice_and_effect_tokens
)
from .audio_utils import mix_with_background, convert_to_mp3
from .constants import AUDIO_EFFECTS
import re
from importlib import resources
import logging
import json
import tempfile
import torch
from kokoro_matrix.constants import USER_VOICES_DIR

logging.basicConfig(filename='audio_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

stop_spinner = False
stop_audio = False

from pydub import AudioSegment

class AudioProcessor:
    """Handles processing of long audio outputs with progress tracking."""

    def __init__(self, pipeline, base_output_dir: str = None):
        self.pipeline = pipeline
        self.base_output_dir = base_output_dir or os.path.join(os.getcwd(), "temp_audio")
        self.current_segment = 0
        self.total_segments = 0
        self._stop_processing = False
        self._progress = 0
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set a callback function to track progress."""
        self._progress_callback = callback

    def create_temp_directory(self) -> str:
        """Create a temporary directory for chunk storage."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = os.path.join(self.base_output_dir, f"temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def process_chunk(self, chunk: str, voice: str, speed: float, chunk_num: int,
                     temp_dir: str) -> Optional[str]:
        """Process a single chunk and save to temporary file."""
        try:
            samples, sr = process_chunk_sequential(chunk, self.pipeline, voice, speed)
            if samples is not None:
                temp_file = os.path.join(temp_dir, f"chunk_{chunk_num:04d}.wav")
                sf.write(temp_file, samples, sr)
                return temp_file
        except Exception as e:
            print(f"\nError processing chunk {chunk_num}: {e}")
        return None

    def merge_chunks(self, chunk_files: List[str], output_file: str,
            sample_rate: int = 24000, chunk_gap: float = 0.5,
            batch_size: int = 100, background_audio: str = None,
            background_volume: float = 0.3) -> None:
        logging.info(f"Merging {len(chunk_files)} chunks")
        """Merge audio chunks into final output file, processing in batches."""
        if not chunk_files:
            logging.error("No chunks to merge!")
            return
        missing_files = [f for f in chunk_files if not os.path.exists(f)]
        if missing_files:
            logging.error(f"Missing chunk files: {missing_files}")
            raise FileNotFoundError(f"Missing {len(missing_files)} chunk files")

        for chunk_file in chunk_files:
            file_size = os.path.getsize(chunk_file)
            logging.debug(f"Chunk file {chunk_file} size: {file_size} bytes")

        logging.debug(f"Starting merge of {len(chunk_files)} chunks")
        print("\nMerging audio chunks...")

        if os.path.exists(output_file):
            os.remove(output_file)

        current_batch = np.array([], dtype=np.float32)
        batch_count = 0

        for chunk_idx, chunk_file in enumerate(tqdm(chunk_files, desc="Merging chunks")):
            logging.debug(f"Processing chunk file {chunk_idx}: {chunk_file}")
            if self._stop_processing:
                break

            try:
                data, sr = sf.read(chunk_file)
                logging.debug(f"Read chunk file: shape={data.shape}, sr={sr}")
                if sr != sample_rate:
                    print(f"Warning: Sample rate mismatch in {chunk_file}")
                    continue

                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                data = np.array(data, dtype=np.float32)
                silence = np.zeros(int(chunk_gap * sr), dtype=np.float32)

                if current_batch.size == 0:
                    current_batch = data
                else:
                    current_batch = np.concatenate([current_batch, data])
                current_batch = np.concatenate([current_batch, silence])
                if len(current_batch) >= batch_size * sample_rate or chunk_idx == len(chunk_files) - 1:
                    if batch_count == 0:
                        sf.write(output_file, current_batch, sample_rate)
                    else:
                        with sf.SoundFile(output_file, 'r+') as f:
                            f.seek(0, sf.SEEK_END)
                            f.write(current_batch)

                    current_batch = np.array([], dtype=np.float32)  # Clear batch
                    batch_count += 1

            except Exception as e:
                print(f"Error processing {chunk_file}: {e}")
                continue

        print("\nMerged audio chunks successfully... Processing output...")

        if background_audio and os.path.exists(background_audio):
            try:
                print("\nAdding background audio...")
                mixed_output = mix_with_background(
                    output_file,
                    background_audio,
                    background_volume
                )
                if mixed_output != output_file:
                    import shutil
                    shutil.move(mixed_output, output_file)
            except Exception as e:
                print(f"Error adding background audio: {e}")

        print("\nProcessing almost complete, finalizing the output! ✨")

    def process_with_voice_changes(self, segments: List[VoiceSegment],
                         output_file: str, speed: float = 1.0,
                         chunk_gap: float = 0,
                         voice_gap: float = 0,
                         background_audio: str = None,
                         background_volume: float = 0.3,
                         use_goldilocks: bool = True) -> Optional[str]:
        """Process text with different voices and effects."""
        temp_dir = self.create_temp_directory()
        chunk_files = []
        processed_chunks = 0
        result = None

        total_chunks = sum(len(chunk_text(seg.text, chunk_size=2000, use_goldilocks=use_goldilocks))
                              for seg in segments if not seg.is_effect)

        logging.info(f"Total chunks to process: {total_chunks}")
        self.total_segments = len(segments)
        logging.debug(f"Starting to process {len(segments)} segments")

        try:
            for i, segment in enumerate(segments, 1):
                self.current_segment = i
                if self._progress_callback:
                    self._progress_callback(i, len(segments))

                if not segment.is_effect:
                    if segment.voice not in self.pipeline.voices:
                        user_voice_path = os.path.join(USER_VOICES_DIR, f"{segment.voice}.pt")
                        if os.path.exists(user_voice_path):
                            try:
                                logging.debug(f"Loading user voice from token: {segment.voice}")
                                voice_tensor = torch.load(user_voice_path, weights_only=True)
                                self.pipeline.voices[segment.voice] = voice_tensor
                                logging.debug(f"Successfully loaded user voice tensor for: {segment.voice}")
                            except Exception as e:
                                logging.error(f"Error loading user voice '{segment.voice}': {e}")
                    logging.debug(f"Processing text segment {i} with voice {segment.voice}")
                    chunks = chunk_text(segment.text, chunk_size=2000, use_goldilocks=use_goldilocks)
                    logging.info(f"Processing segment {i}/{len(segments)} with {len(chunks)} chunks")

                    for chunk_num, chunk in enumerate(chunks):
                        chunk_file = os.path.join(temp_dir, f"chunk_{len(chunk_files):04d}.wav")
                        logging.debug(f"Processing chunk to file: {chunk_file}")

                        try:
                            samples, sr = process_chunk_sequential(chunk, self.pipeline, segment.voice, speed)
                            if samples is None:
                                logging.error(f"Failed to process chunk {chunk_num} of segment {i}")
                                continue

                            sf.write(chunk_file, samples, sr)
                            chunk_files.append(chunk_file)
                            processed_chunks += 1

                            logging.debug(f"Successfully wrote chunk to: {chunk_file}")
                            logging.info(f"Processed chunk {processed_chunks}/{total_chunks}")

                            if chunk_num < len(chunks) - 1:
                                silence_file = os.path.join(temp_dir, f"silence_chunk_{len(chunk_files):04d}.wav")
                                logging.debug(f"Adding silence chunk between chunks: {silence_file}")
                                silence = np.zeros(int(chunk_gap * 24000), dtype=samples.dtype)
                                sf.write(silence_file, silence, sr)
                                chunk_files.append(silence_file)
                                logging.debug(f"Successfully wrote chunk silence: {silence_file}")

                        except Exception as e:
                            logging.error(f"Error processing chunk {chunk_num} of segment {i}: {e}")
                            raise

                    if i < len(segments):
                        silence_file = os.path.join(temp_dir, f"silence_seg_{len(chunk_files):04d}.wav")
                        logging.debug(f"Adding voice gap after segment {i}: {silence_file}")
                        silence = np.zeros(int(voice_gap * 24000), dtype=np.float32)
                        sf.write(silence_file, silence, 24000)
                        chunk_files.append(silence_file)
                        logging.debug(f"Successfully wrote voice gap: {silence_file}")

                elif segment.is_effect:
                    logging.debug(f"Processing effect segment {i}: {segment.effect_type}")
                    if segment.effect_type == 'silence':
                        silence_file = os.path.join(temp_dir, f"silence_{len(chunk_files):04d}.wav")
                        logging.debug(f"Creating silence effect: {silence_file}")
                        silence = np.zeros(int(segment.effect_duration * 24000), dtype=np.float32)
                        sf.write(silence_file, silence, 24000)
                        chunk_files.append(silence_file)
                        logging.debug(f"Added silence effect file: {silence_file}")
                    else:
                        effect_path = AUDIO_EFFECTS.get(segment.effect_type)
                        logging.debug(f"Effect path for {segment.effect_type}: {effect_path}")
                        if effect_path and os.path.exists(effect_path):
                            effect_file = os.path.join(temp_dir, f"effect_{len(chunk_files):04d}.wav")
                            logging.debug(f"Loading effect audio from {effect_path}")
                            effect_audio = AudioSegment.from_wav(effect_path)
                            effect_audio = effect_audio.set_frame_rate(24000)
                            effect_audio = effect_audio - (20 * (1 - (segment.effect_volume / 100.0)))
                            effect_audio.export(effect_file, format='wav')
                            chunk_files.append(effect_file)
                            logging.debug(f"Added effect file: {effect_file}")
                        else:
                            logging.error(f"Effect path not found: {effect_path}")

            if chunk_files and not self._stop_processing:
                logging.info(f"Processing completed. Processed {processed_chunks}/{total_chunks} chunks")
                self.merge_chunks(
                    chunk_files,
                    output_file,
                    chunk_gap=chunk_gap,
                    background_audio=background_audio,
                    background_volume=background_volume
                )
                result = output_file

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise

        finally:
            logging.debug(f"Final chunk files count: {len(chunk_files)}")
            if os.path.exists(temp_dir):
                for chunk_file in chunk_files:
                    try:
                        os.remove(chunk_file)
                        logging.debug(f"Cleaned up temporary file: {chunk_file}")
                    except Exception as e:
                        logging.error(f"Failed to remove temporary file {chunk_file}: {e}")
                try:
                    os.rmdir(temp_dir)
                    logging.debug(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    logging.error(f"Failed to remove temporary directory {temp_dir}: {e}")
            return result

    def stop_processing(self) -> None:
        """Stop current processing."""
        self._stop_processing = True

    def get_progress(self) -> Tuple[int, int]:
        """Get current processing progress."""
        return self.current_segment, self.total_segments

def process_chunk_sequential(chunk: str, pipeline, voice: str, speed: float) -> tuple:
    try:
        audio_segments = []
        for _, _, audio in pipeline(chunk, voice=voice, speed=speed):
            if audio is not None:
                audio_segments.append(audio.numpy())
        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            return combined_audio, 24000
        return None, None
    except Exception as e:
        print(f"\nError processing chunk: {e}")
        return None, None

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by:
    - Converting to lowercase
    - Replacing spaces with underscores
    - Removing special characters
    - Ensuring it's a valid filename

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename
    """
    filename = filename.lower()
    filename = filename.replace(' ', '_')

    import re
    filename = re.sub(r'[^\w\-\.]', '', filename)

    if not filename:
        import time
        filename = f"audio_{int(time.time())}"

    max_length = 100
    if len(filename) > max_length:
        parts = filename.rsplit('.', 1)
        if len(parts) > 1:
            filename = parts[0][:max_length-len(parts[1])-1] + '.' + parts[1]
        else:
            filename = filename[:max_length]

    return filename

def add_post_split_silence(audio_file: str, silence_duration: float) -> None:
    """
    Add silence to the end of an audio file.

    Args:
        audio_file: Path to the audio file
        silence_duration: Duration of silence in seconds
    """
    if silence_duration <= 0 or not os.path.exists(audio_file):
        return

    try:
        audio_data, sr = sf.read(audio_file)
        silence = np.zeros(int(silence_duration * sr), dtype=audio_data.dtype)
        combined_audio = np.concatenate([audio_data, silence])
        sf.write(audio_file, combined_audio, sr)
        logging.debug(f"Added {silence_duration}s silence to the end of {audio_file}")
    except Exception as e:
        logging.error(f"Error adding post-split silence to {audio_file}: {e}")

def process_output_file(result, output_path, format, post_split_gap):
    """
    Process the output file: add post-split silence, convert to MP3 if needed.

    Args:
        result: Path to the result file (WAV)
        output_path: Target output path
        format: Output format (wav/mp3)
        post_split_gap: Silence to add at the end

    Returns:
        Path to the final output file
    """
    if not result or not os.path.exists(result):
        return None

    if post_split_gap > 0:
        add_post_split_silence(result, post_split_gap)

    if format == "mp3":
        try:
            convert_to_mp3(result, output_path)
            os.remove(result)
            print(f"\nCreated audio: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error converting to MP3: {e}")
            return result

    print(f"\nCreated audio: {result}")
    return result

def convert_text_to_audio(input_file, output_file=None, output_dir=None, voice=None,
                         speed=1.0, lang="a", format="wav", split_output=False,
                         section_pattern=None, spoken_patterns=None, silent_patterns=None,
                         spoken_silence=1.0, silent_duration=1.0, chunk_gap=0, voice_gap=0,
                         section_gap=0, background_audio=None, background_volume=0.3,
                         use_goldilocks=True, convert_caps=False, post_split_gap=0.5):
    """
    Convert text to audio with options for splitting into multiple files.

    When split_output is True:
    - Returns a list of output files
    - Ignores output_file parameter and uses output_dir
    - Creates separate files for chapters, sections, or paragraphs

    When split_output is False:
    - Returns a single output file path
    - Creates one audio file for the entire input
    """
    audio_config = AudioConfig(
        speed=speed,
        format=format,
        chunk_gap=chunk_gap,
        voice_gap=voice_gap,
        section_gap=section_gap,
        use_goldilocks=use_goldilocks,
        convert_caps=convert_caps,
        post_split_gap=post_split_gap
    )

    pattern_config = PatternConfig(
        section_pattern=section_pattern,
        spoken_patterns=spoken_patterns,
        silent_patterns=silent_patterns,
        spoken_silence=spoken_silence,
        silent_duration=silent_duration
    )

    bg_config = BackgroundAudioConfig(
        audio_file=background_audio,
        volume=background_volume
    )

    converter = TextToAudioConverter(lang_code=lang, base_output_dir=output_dir)
    return converter.process_file(
        input_file, output_file, output_dir, voice,
        audio_config, pattern_config, bg_config, split_output
    )

def process_text_with_patterns(
    text: str,
    spoken_patterns: Optional[Dict[str, str]],
    silent_patterns: Optional[Dict[str, str]],
    spoken_silence: float,
    silent_duration: float,
    default_voice: str
) -> List[VoiceSegment]:
    """Process text and create voice segments with silence for patterns."""
    if not spoken_patterns and not silent_patterns:
        return parse_voice_and_effect_tokens(text, default_voice=default_voice)

    base_segments = parse_voice_and_effect_tokens(text, default_voice=default_voice)

    final_segments = []

    spoken_regexes = {k: re.compile(v, re.MULTILINE | re.IGNORECASE) for k, v in (spoken_patterns or {}).items()}
    silent_regexes = {k: re.compile(v, re.MULTILINE | re.IGNORECASE) for k, v in (silent_patterns or {}).items()}

    for segment in base_segments:
        if segment.is_effect:
            final_segments.append(segment)
            continue

        current_text = segment.text
        current_voice = segment.voice

        matches = []
        for name, regex in spoken_regexes.items():
            for match in regex.finditer(current_text):
                matches.append(('spoken', name, match))

        logging.debug(f"Processing text with patterns:")
        logging.debug(f"Silent patterns: {silent_patterns}")
        for name, regex in silent_regexes.items():
            silent_matches = list(regex.finditer(current_text))
            logging.debug(f"Pattern '{name}' found {len(silent_matches)} matches")
            for match in silent_matches:
                logging.debug(f"Match: '{match.group(0)}'")
                matches.append(('silent', name, match))

        matches.sort(key=lambda x: x[2].start())

        last_end = 0
        for match_type, name, match in matches:
            if match.start() > last_end:
                pre_text = current_text[last_end:match.start()]
                if pre_text.strip():
                    final_segments.append(VoiceSegment(
                        text=pre_text,
                        voice=current_voice
                    ))

            if match_type == 'spoken':
                final_segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    is_effect=True,
                    effect_type='silence',
                    effect_duration=spoken_silence
                ))
                final_segments.append(VoiceSegment(
                    text=match.group(0),
                    voice=current_voice
                ))
                final_segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    is_effect=True,
                    effect_type='silence',
                    effect_duration=spoken_silence
                ))
            else:  # silent
                final_segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    is_effect=True,
                    effect_type='silence',
                    effect_duration=silent_duration
                ))

            last_end = match.end()

        if last_end < len(current_text):
            remaining = current_text[last_end:]
            if remaining.strip():
                final_segments.append(VoiceSegment(
                    text=remaining,
                    voice=current_voice
                ))
    logging.debug("Sorted matches:")
    for match_type, name, match in matches:
        logging.debug(f"{match_type}: {name} at position {match.start()}-{match.end()}: '{match.group(0)}'")
    return final_segments

def parse_sections(text: str, section_pattern: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Parse text into sections based on a regex pattern.
    If no pattern provided, treats entire text as one section.

    Args:
        text: Input text content
        section_pattern: Regex pattern to match section delimiters.
                        Text between matches will form sections.

    Returns:
        List of dicts with 'title' and 'content' keys
    """
    if not section_pattern:
        return [{'title': 'Section 1', 'content': text}]

    try:
        matches = list(re.finditer(section_pattern, text, re.MULTILINE | re.IGNORECASE))

        if not matches:
            return [{'title': 'Section 1', 'content': text}]

        sections = []

        if matches[0].start() > 0:
            pre_content = text[:matches[0].start()]
            if pre_content.strip():
                sections.append({
                    'title': 'Section 1',
                    'content': pre_content
                })

        for i, match in enumerate(matches):
            match_text = match.group(0).strip()
            section_num = i + len(sections) + 1

            start = match.end()
            end = matches[i + 1].start() if i < len(matches) - 1 else len(text)

            content = text[start:end].strip()

            title = match_text if match_text else f"Section {section_num}"

            if content:
                sections.append({
                    'title': title,
                    'content': content
                })

        if not sections:
            return [{'title': 'Section 1', 'content': text}]

        return sections

    except Exception as e:
        logging.error(f"Error parsing sections: {e}")
        return [{'title': 'Section 1', 'content': text}]

class AudioConfig:
    """Configuration settings for audio processing."""

    def __init__(self,
                 speed=1.0,
                 format="wav",
                 chunk_gap=0,
                 voice_gap=0,
                 section_gap=0,
                 use_goldilocks=True,
                 convert_caps=False,
                 post_split_gap=0.5):
        self.speed = speed
        self.format = format
        self.chunk_gap = chunk_gap
        self.voice_gap = voice_gap
        self.section_gap = section_gap
        self.use_goldilocks = use_goldilocks
        self.convert_caps = convert_caps
        self.post_split_gap = post_split_gap


class BackgroundAudioConfig:
    """Configuration for background audio."""

    def __init__(self, audio_file=None, volume=0.3):
        self.audio_file = audio_file
        self.volume = volume


class PatternConfig:
    """Configuration for text pattern handling."""

    def __init__(self,
                 section_pattern=None,
                 spoken_patterns=None,
                 silent_patterns=None,
                 spoken_silence=1.0,
                 silent_duration=1.0):
        self.section_pattern = section_pattern
        self.spoken_patterns = spoken_patterns or {}
        self.silent_patterns = silent_patterns or {}
        self.spoken_silence = spoken_silence
        self.silent_duration = silent_duration


class TextToAudioConverter:
    """Handles conversion of text to audio with various processing options."""

    def __init__(self, lang_code="a", base_output_dir=None):
        """Initialize the converter with language and output directory."""
        self.lang_code = lang_code
        self.pipeline = None
        self.base_output_dir = base_output_dir or os.path.join(os.getcwd(), "temp_audio")
        self.processor = None

    def initialize(self, voice=None):
        """Initialize the TTS pipeline and set up the audio processor."""
        if self.pipeline is None:
            self.pipeline = initialize_pipeline(lang_code=self.lang_code)

        if voice is None:
            print("\nNo voice provided. Using default voice: af_bella")
            voice = "af_bella"
        else:
            from kokoro_matrix.pipeline import validate_voice
            voice = validate_voice(voice, self.pipeline)

        self.processor = AudioProcessor(self.pipeline, self.base_output_dir)
        return voice

    def process_file(self, input_file, output_file=None, output_dir=None, voice=None,
                    audio_config=None, pattern_config=None, bg_config=None, split_output=False):
        """Process a file and convert to audio with options for splitting."""
        audio_config = audio_config or AudioConfig()
        pattern_config = pattern_config or PatternConfig()
        bg_config = bg_config or BackgroundAudioConfig()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if not output_file:
                base_name = os.path.basename(input_file)
                base, _ = os.path.splitext(base_name)
                base = sanitize_filename(base)
                output_file = os.path.join(output_dir, f"{base}.{audio_config.format}")
            elif not os.path.isabs(output_file) and not output_file.startswith(output_dir):
                output_file = os.path.join(output_dir, os.path.basename(output_file))
        elif not output_file:
            base, _ = os.path.splitext(input_file)
            output_file = f"{base}.{audio_config.format}"

        voice = self.initialize(voice)

        if split_output or pattern_config.section_pattern:
            return self._process_split_output(
                input_file, output_dir or os.path.dirname(output_file),
                voice, audio_config, pattern_config, bg_config
            )
        else:
            return self._process_single_output(
                input_file, output_file, voice, audio_config, pattern_config, bg_config
            )

    def _process_single_output(self, input_file, output_file, voice,
                              audio_config, pattern_config, bg_config):
        """Process a file into a single audio output."""
        all_segments = self._prepare_segments_from_file(
            input_file, voice, pattern_config, audio_config, audio_config.convert_caps
        )

        if audio_config.format == "mp3":
            wav_output = output_file.replace('.mp3', '.wav')
            if os.path.exists(wav_output):
                os.remove(wav_output)
            if os.path.exists(output_file):
                os.remove(output_file)
        else:
            wav_output = output_file

        result = self.processor.process_with_voice_changes(
            all_segments, wav_output, audio_config.speed,
            chunk_gap=audio_config.chunk_gap,
            voice_gap=audio_config.voice_gap,
            background_audio=bg_config.audio_file,
            background_volume=bg_config.volume,
            use_goldilocks=audio_config.use_goldilocks
        )

        return self._process_output_file(result, output_file, audio_config.format, 0)

    def _process_split_output(self, input_file, output_dir, voice,
                             audio_config, pattern_config, bg_config):
        """Process a file into multiple audio outputs based on sections."""
        os.makedirs(output_dir, exist_ok=True)

        if input_file.endswith('.epub'):
            return self._process_epub_file(
                input_file, output_dir, voice, audio_config, pattern_config, bg_config
            )
        else:
            return self._process_text_file(
                input_file, output_dir, voice, audio_config, pattern_config, bg_config
            )

    def _process_epub_file(self, input_file, output_dir, voice, audio_config, pattern_config, bg_config):
        """Process an EPUB file, creating audio for each chapter."""
        chapters = extract_chapters_from_epub(input_file)
        if not chapters:
            print("No chapters found in EPUB file.")
            return []

        print(f"\nProcessing EPUB with {len(chapters)} chapters...")

        output_files = []

        for i, chapter in enumerate(chapters, 1):
            chapter_title = chapter['title'].replace('/', '_').replace('\\', '_')
            chapter_content = chapter['content']

            if audio_config.convert_caps:
                from kokoro_matrix.text_processing import convert_all_caps
                chapter_content = convert_all_caps(chapter_content)

            base_name = os.path.basename(input_file).rsplit('.', 1)[0]
            sanitized_base = sanitize_filename(base_name)
            sanitized_title = sanitize_filename(chapter_title)
            chapter_filename = f"{sanitized_base}_chapter_{i:03d}_{sanitized_title}.{audio_config.format}"
            output_path = os.path.join(output_dir, chapter_filename)

            print(f"\nProcessing Chapter {i}/{len(chapters)}: {chapter_title}")

            segments = self._create_voice_segments(
                chapter_content, voice, pattern_config
            )

            if audio_config.format == "mp3":
                wav_output = output_path.replace('.mp3', '.wav')
            else:
                wav_output = output_path

            if os.path.exists(wav_output):
                os.remove(wav_output)

            result = self.processor.process_with_voice_changes(
                segments, wav_output, audio_config.speed,
                chunk_gap=audio_config.chunk_gap,
                voice_gap=audio_config.voice_gap,
                background_audio=bg_config.audio_file,
                background_volume=bg_config.volume,
                use_goldilocks=audio_config.use_goldilocks
            )

            processed_file = self._process_output_file(result, output_path, audio_config.format, audio_config.post_split_gap)
            if processed_file:
                output_files.append(processed_file)

        return output_files

    def _process_text_file(self, input_file, output_dir, voice, audio_config, pattern_config, bg_config):
        """Process a text file, splitting into sections if needed."""
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

            if audio_config.convert_caps:
                from kokoro_matrix.text_processing import convert_all_caps
                text = convert_all_caps(text)

            output_files = []

            if pattern_config.section_pattern:
                sections = parse_sections(text, pattern_config.section_pattern)
                print(f"\nFound {len(sections)} sections using pattern: {pattern_config.section_pattern}")

                for i, section in enumerate(sections, 1):
                    output_files.extend(self._process_section(
                        section, i, len(sections), input_file, output_dir, voice,
                        audio_config, pattern_config, bg_config
                    ))
            else:
                output_files.extend(self._process_paragraphs(
                    text, input_file, output_dir, voice, audio_config, pattern_config, bg_config
                ))

            return output_files

    def _process_section(self, section, index, total_sections, input_file, output_dir,
                        voice, audio_config, pattern_config, bg_config):
        """Process a single section from a text file."""
        section_title = section['title'].replace('/', '_').replace('\\', '_')
        section_content = section['content']

        base_name = os.path.basename(input_file).rsplit('.', 1)[0]
        sanitized_base = sanitize_filename(base_name)
        sanitized_title = sanitize_filename(section_title)
        section_filename = f"{sanitized_base}_section_{index:03d}_{sanitized_title[:50]}.{audio_config.format}"
        output_path = os.path.join(output_dir, section_filename)

        print(f"\nProcessing Section {index}/{total_sections}: {section_title}")

        segments = self._create_voice_segments(
            section_content, voice, pattern_config
        )

        if audio_config.format == "mp3":
            wav_output = output_path.replace('.mp3', '.wav')
        else:
            wav_output = output_path

        if os.path.exists(wav_output):
            os.remove(wav_output)

        result = self.processor.process_with_voice_changes(
            segments, wav_output, audio_config.speed,
            chunk_gap=audio_config.chunk_gap,
            voice_gap=audio_config.voice_gap,
            background_audio=bg_config.audio_file,
            background_volume=bg_config.volume,
            use_goldilocks=audio_config.use_goldilocks
        )

        processed_file = self._process_output_file(result, output_path, audio_config.format, audio_config.post_split_gap)
        if processed_file:
            print(f"\nCreated section audio: {processed_file}")
            return [processed_file]
        return []

    def _process_paragraphs(self, text, input_file, output_dir, voice, audio_config, pattern_config, bg_config):
        """Process text as individual paragraphs or a single file."""
        paragraphs = re.split(r'\n\s*\n', text)
        filtered_paragraphs = self._filter_paragraphs(paragraphs)

        output_files = []

        if len(filtered_paragraphs) > 1:
            print(f"\nSplitting text into {len(filtered_paragraphs)} paragraphs")

            for i, paragraph in enumerate(filtered_paragraphs, 1):
                if not paragraph.strip():
                    continue

                temp_file = self._create_temp_paragraph_file(paragraph)

                try:
                    base_name = os.path.basename(input_file).rsplit('.', 1)[0]
                    sanitized_base = sanitize_filename(base_name)
                    paragraph_preview = paragraph.strip()[:30].replace('\n', ' ')
                    sanitized_preview = sanitize_filename(paragraph_preview)
                    para_filename = f"{sanitized_base}_para_{i:03d}_{sanitized_preview}.{audio_config.format}"
                    output_path = os.path.join(output_dir, para_filename)

                    print(f"\nProcessing Paragraph {i}/{len(filtered_paragraphs)}")

                    paragraph_audio_config = AudioConfig(
                        speed=audio_config.speed,
                        format=audio_config.format,
                        chunk_gap=audio_config.chunk_gap,
                        voice_gap=audio_config.voice_gap,
                        section_gap=0,
                        use_goldilocks=audio_config.use_goldilocks,
                        convert_caps=audio_config.convert_caps,
                        post_split_gap=audio_config.post_split_gap
                    )

                    segments = self._prepare_segments_from_file(
                        temp_file.name, voice, pattern_config, paragraph_audio_config, audio_config.convert_caps
                    )

                    if audio_config.format == "mp3":
                        wav_output = output_path.replace('.mp3', '.wav')
                    else:
                        wav_output = output_path

                    if os.path.exists(wav_output):
                        os.remove(wav_output)

                    result = self.processor.process_with_voice_changes(
                        segments, wav_output, paragraph_audio_config.speed,
                        chunk_gap=paragraph_audio_config.chunk_gap,
                        voice_gap=paragraph_audio_config.voice_gap,
                        background_audio=bg_config.audio_file,
                        background_volume=bg_config.volume,
                        use_goldilocks=paragraph_audio_config.use_goldilocks
                    )

                    processed_file = self._process_output_file(
                        result, output_path, paragraph_audio_config.format,
                        paragraph_audio_config.post_split_gap
                    )

                    if processed_file:
                        output_files.append(processed_file)
                        print(f"\nCreated paragraph audio: {processed_file}")
                finally:
                    self._cleanup_temp_file(temp_file)

        else:
            base_name = os.path.basename(input_file).rsplit('.', 1)[0]
            sanitized_base = sanitize_filename(base_name)
            output_path = os.path.join(output_dir, f"{sanitized_base}.{audio_config.format}")

            segments = self._prepare_segments_from_file(
                input_file, voice, pattern_config, audio_config, audio_config.convert_caps
            )

            if audio_config.format == "mp3":
                wav_output = output_path.replace('.mp3', '.wav')
            else:
                wav_output = output_path

            if os.path.exists(wav_output):
                os.remove(wav_output)

            result = self.processor.process_with_voice_changes(
                segments, wav_output, audio_config.speed,
                chunk_gap=audio_config.chunk_gap,
                voice_gap=audio_config.voice_gap,
                background_audio=bg_config.audio_file,
                background_volume=bg_config.volume,
                use_goldilocks=audio_config.use_goldilocks
            )

            processed_file = self._process_output_file(
                result, output_path, audio_config.format, audio_config.post_split_gap
            )

            if processed_file:
                output_files.append(processed_file)

        return output_files

    def _create_temp_paragraph_file(self, paragraph):
        """Create a temporary file containing a paragraph."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        temp_file.write(paragraph)
        temp_file.flush()
        return temp_file

    def _cleanup_temp_file(self, temp_file):
        """Clean up a temporary file."""
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def _filter_paragraphs(self, paragraphs):
        """Filter paragraphs to remove dividers and empty lines."""
        filtered_paragraphs = []

        silent_marker_patterns = [
            r'^\s*(?:\*\s*){3,}\s*$',  # Asterisk dividers (***)
            r'^\s*(?:-\s*){3,}\s*$',   # Dash dividers (---)
            r'^\s*(?:=\s*){3,}\s*$',   # Equal sign dividers (===)
            r'^\s*(?:#\s*){3,}\s*$',   # Hash dividers (###)
            r'^\s*$'                   # Empty lines
        ]
        combined_pattern = '|'.join(f'({pattern})' for pattern in silent_marker_patterns)
        silent_regex = re.compile(combined_pattern, re.MULTILINE)

        for para in paragraphs:
            para = para.strip()
            if para and not silent_regex.fullmatch(para):
                filtered_paragraphs.append(para)

        return filtered_paragraphs

    def _prepare_segments_from_file(self, input_file, voice, pattern_config, audio_config, convert_caps):
        """Prepare voice segments from a file."""
        if input_file.endswith('.epub'):
            from kokoro_matrix.text_processing import extract_chapters_from_epub
            chapters = extract_chapters_from_epub(input_file)
            if not chapters:
                print("No chapters found in EPUB file.")
                return []
        else:
            with open(input_file, 'r', encoding='utf-8') as file:
                text = file.read()
                if convert_caps:
                    from kokoro_matrix.text_processing import convert_all_caps
                    text = convert_all_caps(text)
                chapters = [{'title': 'Section 1', 'content': text}]

        return self._create_segments_from_chapters(
            chapters, voice, pattern_config, audio_config
        )

    def _create_segments_from_chapters(self, chapters, voice, pattern_config, audio_config):
        """Create voice segments from chapters."""
        all_segments = []

        for chapter in chapters:
            chapter_segments = self._create_voice_segments(
                chapter['content'], voice, pattern_config
            )

            all_segments.extend(chapter_segments)

            if chapter != chapters[-1] and audio_config.section_gap > 0:
                all_segments.append(VoiceSegment(
                    text="",
                    voice=voice,
                    is_effect=True,
                    effect_type='silence',
                    effect_duration=audio_config.section_gap
                ))

        return all_segments

    def _create_voice_segments(self, text, voice, pattern_config):
        """Create voice segments from text with pattern handling."""
        return process_text_with_patterns(
            text,
            pattern_config.spoken_patterns,
            pattern_config.silent_patterns,
            pattern_config.spoken_silence,
            pattern_config.silent_duration,
            default_voice=voice
        )

    def _process_output_file(self, result, output_path, format, post_split_gap):
        """Process the output file by adding silence and converting format."""
        if not result or not os.path.exists(result):
            return None

        if post_split_gap > 0:
            add_post_split_silence(result, post_split_gap)

        if format == "mp3":
            try:
                convert_to_mp3(result, output_path)
                os.remove(result)
                print(f"\nCreated audio: {output_path}")
                return output_path
            except Exception as e:
                print(f"Error converting to MP3: {e}")
                return result

        print(f"\nCreated audio: {result}")
        return result