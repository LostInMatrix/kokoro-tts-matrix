import gradio as gr
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict, Optional, Tuple, Union
from kokoro_matrix.text_processing import extract_text_from_epub
from kokoro_matrix.pipeline import initialize_pipeline
from kokoro_matrix.audio_processing import (
    convert_text_to_audio,
    AudioProcessor,
    AudioConfig,
    PatternConfig,
    BackgroundAudioConfig
)
from kokoro_matrix.audio_utils import mix_with_background
import logging
import os
import torch
from datetime import datetime
from kokoro_matrix.voice_transformations import add_voice_variation, generate_voice_sample, save_voice
from kokoro_matrix.constants import USER_VOICES_DIR
from kokoro_matrix.help_content import GENERAL_HELP

logging.basicConfig(filename='gradio_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs(USER_VOICES_DIR, exist_ok=True)

LANGUAGE_CODES = {
    "American English": "a",
    "British English": "b",
    "Japanese": "j",
    "Mandarin Chinese": "z",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Brazilian Portuguese": "p",
    "User Blended Voices": "u"
}

MAIN_LANGUAGE_CODES = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Brazilian Portuguese": "p",
    "User Blended Voices": "u"
}

VOICES_BY_LANGUAGE = {
    "American English": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
        "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"
    ],
    "British English": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
    ],
    "Japanese": [
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"
    ],
    "Mandarin Chinese": [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
    ],
    "Spanish": ["ef_dora", "em_alex", "em_santa"],
    "French": ["ff_siwis"],
    "Hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "Italian": ["if_sara", "im_nicola"],
    "Brazilian Portuguese": ["pf_dora", "pm_alex", "pm_santa"]
}

def clear_logs():
    """Clear both log files."""
    log_files = ['audio_debug.log', 'gradio_debug.log']
    for log_file in log_files:
        try:
            with open(log_file, 'w') as f:
                pass
            logging.debug("Log file cleared")
        except Exception as e:
            return f"Error clearing {log_file}: {str(e)}"
    return "Logs cleared successfully"

def load_logs():
    """Load and format both log files with timestamps."""
    logs = []
    log_files = ['audio_debug.log', 'gradio_debug.log']

    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs.append(f"=== {log_file} ===\n")
                    logs.extend(f.readlines()[-1000:])
            except Exception as e:
                logs.append(f"Error reading {log_file}: {str(e)}\n")
        else:
            logs.append(f"Log file not found: {log_file}\n")

    return "".join(logs)

def update_saved_file_visibility(message, file_path):
    return gr.update(value=file_path, visible=file_path is not None)

def estimate_audio_duration(text: str, speed: float) -> float:
    """
    Estimate audio duration in seconds based on text length and speed.
    Average speaking rate is roughly 150 words per minute.
    """
    words = text.split()
    words_per_minute = 150 * speed  # Adjust for speech speed
    estimated_minutes = len(words) / words_per_minute
    return estimated_minutes * 60  # Convert to seconds

def update_voices(language: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Update voice choices based on selected language."""
    voices = VOICES_BY_LANGUAGE.get(language, [])
    default_value = voices[0] if voices else None

    return (
        gr.Dropdown(choices=voices, value=default_value),  # voice
        gr.Dropdown(choices=voices, value=default_value),  # base_voice
        gr.Dropdown(choices=voices, value=default_value),  # secondary_voice
        gr.Dropdown(choices=voices, value=default_value)   # tertiary_voice
    )
def scan_user_voices():
    """Scan the user voices directory for .pt files and return their names."""
    voices = []
    if os.path.exists(USER_VOICES_DIR):
        for file in os.listdir(USER_VOICES_DIR):
            if file.endswith('.pt'):
                voice_name = os.path.splitext(file)[0]
                voices.append(voice_name)
    return voices

def update_voice_list():
    global VOICES_BY_LANGUAGE
    VOICES_BY_LANGUAGE["User Blended Voices"] = scan_user_voices()

def read_file_preview(file) -> str:
    """Read and return a preview of the file content."""
    if file is None:
        return ""
    try:
        if hasattr(file, 'name'):
            if file.name.endswith('.epub'):
                text = extract_text_from_epub(file.name)
            else:
                with open(file.name, 'r', encoding='utf-8') as f:
                    text = f.read()
            words = text.split()
            preview = " ".join(words[:500])
            return f"{preview}...\n\nTotal words: {len(words)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return ""

def get_voice_from_selection(language, voice_name):
    """Get the voice based on language and voice name."""
    if language == "User Blended Voices":
        voice_path = os.path.join(USER_VOICES_DIR, f"{voice_name}.pt")
        if os.path.exists(voice_path):
            try:
                return torch.load(voice_path, weights_only=True)
            except Exception as e:
                print(f"Error loading voice {voice_name}: {str(e)}")
                return None

    return voice_name

def generate_phoneme_example(phoneme_text, voice):
    try:
        from kokoro_matrix.pipeline import initialize_pipeline
        import tempfile
        import torch
        import soundfile as sf
        import logging

        logging.debug(f"Generating phoneme example with voice: {voice}")
        logging.debug(f"Input text: {phoneme_text}")

        pipeline = initialize_pipeline("a")

        audio_segments = []

        try:
            for _, _, audio in pipeline(phoneme_text, voice=voice, speed=1.0):
                if audio is not None:
                    audio_segments.append(audio.numpy())

            if not audio_segments:
                logging.error("No audio segments generated")
                return None

            combined_audio = np.concatenate(audio_segments)

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            sf.write(temp_path, combined_audio, 24000)
            logging.debug(f"Saved audio to {temp_path}")

            return temp_path
        except Exception as inner_e:
            logging.error(f"Error in audio generation: {str(inner_e)}")
            return None
    except Exception as e:
        logging.error(f"Error generating phoneme example: {str(e)}")
        print(f"Error generating phoneme example: {e}")
        return None

def upload_voice_file(files):
    """Upload voice files to the user_voices directory."""
    if not files:
        return "No files selected.", []

    results = []
    successful = 0

    for file in files:
        try:
            try:
                voice_tensor = torch.load(file.name, weights_only=True)
                if not isinstance(voice_tensor, torch.Tensor):
                    results.append(f"❌ {os.path.basename(file.name)}: Not a valid voice tensor")
                    continue
            except Exception as e:
                results.append(f"❌ {os.path.basename(file.name)}: {str(e)}")
                continue

            import shutil
            filename = os.path.basename(file.name)
            destination = os.path.join(USER_VOICES_DIR, filename)
            shutil.copy2(file.name, destination)

            results.append(f"✅ {filename}: Successfully uploaded")
            successful += 1

        except Exception as e:
            results.append(f"❌ {os.path.basename(file.name)}: {str(e)}")

    update_voice_list()

    voices = scan_user_voices()
    voices_df = [[voice] for voice in voices]

    message = f"Uploaded {successful} voice file(s) successfully.\n" + "\n".join(results)
    return message, voices_df

def refresh_user_voices():
    """Refresh the list of user voices."""
    update_voice_list()
    voices = scan_user_voices()
    voices_df = [[voice] for voice in voices]
    return voices_df

def generate_preview(file, text, input_method, preview_type, use_goldilocks=True, chunk_size=2000):
    """Generate different types of text previews based on the selected preview type."""
    preview_text = ""

    if input_method == "Text Input" and text:
        input_text = text
    elif input_method == "File Upload" and file is not None:
        try:
            if file.name.endswith('.epub'):
                input_text = extract_text_from_epub(file.name)
            else:
                with open(file.name, 'r', encoding='utf-8') as f:
                    input_text = f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    else:
        return "No input provided"

    if preview_type == "Basic":
        words = input_text.split()
        preview_text = f"{input_text}\n\nTotal words: {len(words)}"

    elif preview_type == "Chunking Analysis":
        from kokoro_matrix.text_processing import chunk_text
        from kokoro_matrix.goldilocks import GoldilocksChunker

        words = input_text.split()
        preview_text = f"Text Overview:\n"
        preview_text += f"Total words: {len(words)}\n"
        preview_text += f"Total characters: {len(input_text)}\n\n"

        goldilocks_chunker = GoldilocksChunker()
        goldilocks_chunks = chunk_text(input_text, use_goldilocks=True)

        preview_text += f"Goldilocks Chunking (Token-based):\n"
        preview_text += f"Total chunks: {len(goldilocks_chunks)}\n"

        for i, chunk in enumerate(goldilocks_chunks):
            chunk_words = len(chunk.split())
            chunk_chars = len(chunk)
            estimated_tokens = goldilocks_chunker.estimate_tokens(chunk)
            preview_text += f"\nChunk {i+1} ({chunk_words} words, ~{estimated_tokens} tokens):\n"
            preview_text += f"{chunk}\n"

        char_chunks = chunk_text(input_text, chunk_size=chunk_size, use_goldilocks=False)

        preview_text += f"\n\nCharacter-based Chunking (size={chunk_size}):\n"
        preview_text += f"Total chunks: {len(char_chunks)}\n"

        for i, chunk in enumerate(char_chunks):
            chunk_words = len(chunk.split())
            chunk_chars = len(chunk)
            preview_text += f"\nChunk {i+1} ({chunk_words} words, {chunk_chars} chars):\n"
            preview_text += f"{chunk}\n"

    elif preview_type == "Voice Tags":
        from kokoro_matrix.voice_segments import parse_voice_and_effect_tokens

        words = input_text.split()
        preview_text = f"Text Overview:\n"
        preview_text += f"Total words: {len(words)}\n\n"

        segments = parse_voice_and_effect_tokens(input_text, default_voice="af_bella")

        preview_text += f"Voice Analysis:\n"
        preview_text += f"Identified {len(segments)} voice/effect segments\n\n"

        voice_count = {}
        effect_count = {}

        for i, segment in enumerate(segments):
            if segment.is_effect:
                effect_count[segment.effect_type] = effect_count.get(segment.effect_type, 0) + 1
                preview_text += f"Segment {i+1}: Effect [{segment.effect_type}]\n"
            else:
                voice_count[segment.voice] = voice_count.get(segment.voice, 0) + 1
                preview_text += f"Segment {i+1}: Voice [{segment.voice}] - \"{segment.text}\"\n"

        preview_text += "\nVoice Usage Summary:\n"
        for voice, count in voice_count.items():
            preview_text += f"- {voice}: {count} segments\n"

        if effect_count:
            preview_text += "\nEffect Usage Summary:\n"
            for effect, count in effect_count.items():
                preview_text += f"- {effect}: {count} occurrences\n"

    return preview_text

def process_tts(
    file_obj: Optional[str],
    text_input: str,
    language: str,
    voice: str,
    speed: float,
    format: str,
    section_pattern: Optional[str],
    spoken_patterns: list[str],
    silent_patterns: list[str],
    spoken_silence: float,
    silent_duration: float,
    custom_pattern: str,
    custom_pattern_type: str,
    chunk_gap: float,
    voice_gap: float,
    section_gap: float,
    background_audio: Optional[str],
    background_volume: float,
    convert_caps: bool,
    output_dir: str = None,
    enable_split: bool = False,
    split_method: str = "Auto detect",
    post_split_gap: float = 0.5,
) -> Tuple[Dict[str, Union[str, int]], Optional[str]]:
    """Process text-to-speech conversion with progress updates."""

    if file_obj is None and not text_input:
        return ({"status": "error", "message": "Please provide either a file or text input."}, None)

    effective_section_pattern = section_pattern

    try:
        if language == "User Blended Voices":
            try:
                lang_code = voice.split('_')[0]
                if lang_code not in LANGUAGE_CODES.values():
                    lang_code = "a"
            except:
                lang_code = "a"

            voice_path = os.path.join(USER_VOICES_DIR, f"{voice}.pt")
            if os.path.exists(voice_path):
                try:
                    voice = torch.load(voice_path, weights_only=True)
                except Exception as e:
                    return ({"status": "error", "message": f"Error loading voice: {str(e)}"}, None)
        else:
            lang_code = MAIN_LANGUAGE_CODES.get(language, "a")

        pipeline = initialize_pipeline(lang_code)
        processor = AudioProcessor(pipeline)

        if language == "User Blended Voices":
            voice_path = os.path.join(USER_VOICES_DIR, f"{voice}.pt")
            if os.path.exists(voice_path):
                try:
                    voice_tensor = torch.load(voice_path, weights_only=True)
                    pipeline.voices[voice] = voice_tensor
                except Exception as e:
                    return ({"status": "error", "message": f"Error loading voice: {str(e)}"}, None)

        temp_file = None
        if text_input and not file_obj:
            try:
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                temp_file.write(text_input)
                temp_file.flush()
                temp_file.close()
                input_path = temp_file.name
            except Exception as e:
                if temp_file:
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                raise e
        elif file_obj:
            input_path = file_obj.name
        else:
            return ({"status": "error", "message": "No input provided"}, None)

        def progress_callback(current, total):
            if total > 0:
                progress((current / total), desc=f"Processing segment {current}/{total}")

        processor.set_progress_callback(progress_callback)

        selected_spoken, selected_silent = build_section_patterns(
            spoken_patterns,
            silent_patterns,
            custom_pattern,
            custom_pattern_type
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Using output directory: {output_dir}")

        if enable_split:
            try:
                if split_method == "Sections (using pattern)" and not section_pattern.strip():
                    effective_section_pattern = r"^(?:Chapter|CHAPTER)\s+(?:\d+|[A-Za-z]+(?:\s+[A-Za-z]+)?)\s*$"

                audio_config = AudioConfig(
                    speed=speed,
                    format=format,
                    chunk_gap=chunk_gap,
                    voice_gap=voice_gap,
                    section_gap=section_gap,
                    use_goldilocks=True,
                    convert_caps=convert_caps,
                    post_split_gap=post_split_gap
                )

                pattern_config = PatternConfig(
                    section_pattern=effective_section_pattern if effective_section_pattern and effective_section_pattern.strip() else None,
                    spoken_patterns=selected_spoken,
                    silent_patterns=selected_silent,
                    spoken_silence=spoken_silence,
                    silent_duration=silent_duration
                )

                bg_config = BackgroundAudioConfig(
                    audio_file=background_audio,
                    volume=background_volume
                )

                output_files = convert_text_to_audio(
                    input_file=input_path,
                    output_dir=output_dir,
                    voice=voice,
                    speed=audio_config.speed,
                    lang=lang_code,
                    format=audio_config.format,
                    split_output=True,
                    section_pattern=pattern_config.section_pattern,
                    spoken_patterns=pattern_config.spoken_patterns,
                    silent_patterns=pattern_config.silent_patterns,
                    spoken_silence=pattern_config.spoken_silence,
                    silent_duration=pattern_config.silent_duration,
                    chunk_gap=audio_config.chunk_gap,
                    voice_gap=audio_config.voice_gap,
                    section_gap=audio_config.section_gap,
                    background_audio=bg_config.audio_file,
                    background_volume=bg_config.volume,
                    use_goldilocks=audio_config.use_goldilocks,
                    convert_caps=audio_config.convert_caps,
                    post_split_gap=audio_config.post_split_gap
                )

                if output_files and len(output_files) > 0:
                    list_file_path = os.path.join(output_dir, "audio_files_list.txt")
                    with open(list_file_path, "w") as f:
                        f.write(f"Generated {len(output_files)} audio files:\n")
                        for file in output_files:
                            f.write(f"{file}\n")

                    return (
                        {"status": "complete_split",
                         "output_dir": output_dir,
                         "file_count": len(output_files),
                         "files": output_files,
                         "list_file": list_file_path,
                         "progress": 100},
                        list_file_path
                    )
                return (
                    {"status": "error", "message": "Failed to generate split audio files"},
                    None
                )
            except Exception as e:
                logging.error(f"Error in split audio processing: {e}")
                return (
                    {"status": "error", "message": f"Error during split processing: {str(e)}"},
                    None
                )
        try:
            audio_config = AudioConfig(
                speed=speed,
                format=format,
                chunk_gap=chunk_gap,
                voice_gap=voice_gap,
                section_gap=section_gap,
                use_goldilocks=True,
                convert_caps=convert_caps
            )

            pattern_config = PatternConfig(
                section_pattern=section_pattern if section_pattern.strip() else None,
                spoken_patterns=selected_spoken,
                silent_patterns=selected_silent,
                spoken_silence=spoken_silence,
                silent_duration=silent_duration
            )

            bg_config = BackgroundAudioConfig(
                audio_file=background_audio,
                volume=background_volume
            )

            output_file = convert_text_to_audio(
                input_file=input_path,
                output_dir=output_dir,
                voice=voice,
                speed=audio_config.speed,
                lang=lang_code,
                format=audio_config.format,
                split_output=False,
                section_pattern=pattern_config.section_pattern,
                spoken_patterns=pattern_config.spoken_patterns,
                silent_patterns=pattern_config.silent_patterns,
                spoken_silence=pattern_config.spoken_silence,
                silent_duration=pattern_config.silent_duration,
                chunk_gap=audio_config.chunk_gap,
                voice_gap=audio_config.voice_gap,
                section_gap=audio_config.section_gap,
                background_audio=bg_config.audio_file,
                background_volume=bg_config.volume,
                use_goldilocks=audio_config.use_goldilocks,
                convert_caps=audio_config.convert_caps
            )

            if output_file and background_audio and os.path.exists(output_file):
                try:
                    mixed_output = mix_with_background(
                        output_file,
                        background_audio,
                        background_volume
                    )
                    if mixed_output != output_file:
                        output_file = mixed_output
                except Exception as e:
                    print(f"Background mixing failed: {e}")

            if output_file and os.path.exists(output_file):
                return (
                    {"status": "complete", "file": output_file, "progress": 100, "output_dir": output_dir},
                    output_file
                )
            return (
                {"status": "error", "message": "Failed to generate audio"},
                None
            )
        finally:
            if 'input_path' in locals() and os.path.exists(input_path) and input_path != getattr(file_obj, 'name', ''):
                try:
                    os.unlink(input_path)
                except Exception:
                    pass

    except Exception as e:
        return (
            {"status": "error", "message": str(e)},
            None
        )

def update_ui(result) -> Tuple[gr.Audio, gr.File, gr.Textbox]:
    """Update UI elements based on processing result."""
    if result["status"] == "complete":
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(result["file"])
            duration_minutes = len(audio) / (1000 * 60)

            logging.debug(f"Audio duration: {duration_minutes:.2f} minutes")

            if duration_minutes > 2:
                return (
                    gr.Audio(visible=False),
                    gr.File(value=result["file"], label="Download Audio (>2 min, preview disabled)", visible=True, interactive=True),
                    gr.Textbox(value="", visible=False)
                )
        except Exception as e:
            logging.error(f"Error checking audio duration: {e}")

        return (
            gr.Audio(value=result["file"], visible=True),
            gr.File(value=result["file"], label="Download Audio", visible=True, interactive=True),
            gr.Textbox(value="", visible=False)
        )
    elif result["status"] == "complete_split":
        list_file = result.get("list_file")
        file_count = result.get("file_count", 0)
        output_dir = result.get("output_dir", "")

        if list_file and os.path.exists(list_file):
            message = f"Generated {file_count} audio files in directory: {output_dir}"
            return (
                gr.Audio(visible=False),
                gr.File(value=list_file, label=f"Download file list ({file_count} audio files generated)", visible=True, interactive=True),
                gr.Textbox(value=message, label="Split Files Information", visible=True)
            )
        else:
            message = f"Generated {file_count} audio files in directory: {output_dir}"
            return (
                gr.Audio(visible=False),
                gr.File(value=None, label=f"Generated {file_count} audio files", visible=True, interactive=False),
                gr.Textbox(value=message, label="Split Files Information", visible=True)
            )
    elif result["status"] == "error":
        return (
            gr.Audio(visible=False),
            gr.File(value=None, label=f"Error: {result['message']}", visible=True, interactive=False),
            gr.Textbox(value="", visible=False)
        )
    else:
        return (
            gr.Audio(visible=False),
            gr.File(value=None, label=f"Processing: {result.get('progress', 0)}%", visible=True, interactive=False),
            gr.Textbox(value="", visible=False)
        )

def build_section_patterns(spoken_patterns: list, silent_patterns: list,
                         custom_pattern: str, custom_type: str) -> Tuple[dict, dict]:
    """Build regex patterns for spoken and silent markers."""

    spoken_patterns = spoken_patterns or []
    silent_patterns = silent_patterns or []
    custom_pattern = custom_pattern.strip() if custom_pattern else ""

    spoken_map = {
        "Chapter Headers (Chapter X, Chapter One, etc.)":
            r"^(?:Chapter|CHAPTER)\s+(?:\d+|[A-Za-z]+(?:\s+[A-Za-z]+)?)\s*$",
        "Section Headers (Section X)":
            r"^\s*(?:Section|SECTION)\s+(?:\d+|[A-Za-z]+)",
        "Part Headers (Part X, Part One, etc.)":
            r"^\s*(?:Part|PART)\s+(?:\d+|[A-Za-z]+(?:\s+[A-Za-z]+)?)",
        "Numbered Sections (1., 2., etc.)":
            r"^\d+\.\s+",
        "Book Headers (Book X, Book One, etc.)":
            r"^\s*(?:Book|BOOK)\s+(?:\d+|[A-Za-z]+(?:\s+[A-Za-z]+)?)",
        "All Caps Headers (NEW SECTION, etc.)":
            r"^\s*[A-Z][A-Z\s]+[A-Z]\s*$"
    }

    silent_map = {
        "Asterisk Dividers (***)": r"^\s*(?:\*\s*){3,}\s*$",
        "Dash Dividers (---)": r"^\s*(?:-\s*){3,}\s*$",
        "Equal Sign Dividers (===)": r"^\s*(?:=\s*){3,}\s*$",
        "Hash Dividers (###)": r"^\s*(?:#\s*){3,}\s*$",
        "Blank Lines (multiple consecutive empty lines)": r"^\s*$\n^\s*$",
        "Markdown Headers (# Title)": r"^#{1,3}\s+.+$"
    }

    if custom_pattern.strip():
        if custom_type == "Spoken":
            spoken_map["Custom"] = custom_pattern
        else:
            silent_map["Custom"] = custom_pattern

    selected_spoken = {k: v for k, v in spoken_map.items() if k in spoken_patterns}
    selected_silent = {k: v for k, v in silent_map.items() if k in silent_patterns}

    logging.debug(f"Selected spoken patterns: {selected_spoken}")
    logging.debug(f"Selected silent patterns: {selected_silent}")

    return selected_spoken, selected_silent

def launch_gradio():
    """Launch the Gradio interface."""
    with gr.Blocks(title="Kokoro TTS Matrix") as iface:
        with gr.Tabs() as tabs:
            with gr.Tab("Convert Text"):
                gr.Markdown("""
                # Kokoro TTS Matrix
                What if I told you... your text could speak for itself?
                """)

                input_type = gr.Radio(
                    choices=["Text Input", "File Upload"],
                    value="Text Input",
                    label="Choose Input Method"
                )

                input_text = gr.Textbox(
                    label="Enter Text",
                    placeholder="Type or paste your text here...",
                    lines=5,
                    visible=True
                )

                input_file = gr.File(
                    label="Upload Text/EPUB File",
                    file_types=[".txt", ".epub"],
                    file_count="single",
                    visible=False
                )

                with gr.Row():
                    language = gr.Dropdown(
                        choices=list(MAIN_LANGUAGE_CODES.keys()),
                        value="American English",
                        label="Language"
                    )
                    voice = gr.Dropdown(
                        choices=VOICES_BY_LANGUAGE["American English"],
                        value="af_nicole",
                        label="Default Voice"
                    )

                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Speed"
                    )
                    format = gr.Radio(
                        choices=["wav", "mp3"],
                        value="wav",
                        label="Output Format",
                        info="Note: MP3 output requires first creating WAV and then converting, which adds processing time."
                    )

                with gr.Accordion("Text Preview", open=False):
                    text_preview = gr.Textbox(
                        label="Text Preview",
                        lines=10,
                        interactive=False,
                        info="Preview of processed text based on selected preview type"
                    )

                    preview_type = gr.Radio(
                        choices=["Basic", "Chunking Analysis", "Voice Tags"],
                        value="Basic",
                        label="Preview Type"
                    )

                with gr.Row():
                    progress = gr.State(value={"status": "", "progress": 0})
                    with gr.Column():
                        audio_preview = gr.Audio(
                            label="Generated Audio",
                            visible=False,
                            interactive=False,
                            type="filepath"
                        )
                        output_file = gr.File(
                            label="Processing Status",
                            visible=True,
                            interactive=False,
                            type="filepath",
                            elem_id="output_audio"
                        )
                        split_output_info = gr.Textbox(
                            label="Split Audio Information",
                            visible=False,
                            interactive=False
                        )

                convert_btn = gr.Button("Convert to Speech", variant="primary")


            with gr.Tab("Settings"):
                gr.Markdown("""
                # Settings
                Configure some basic options for your text-to-speech conversion.
                """)

                with gr.Accordion("Output Settings", open=True):
                    gr.Markdown("### Output Location")
                    with gr.Row():
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="/app/outputs",
                            value="/app/outputs",
                            info="Files will be saved to this directory"
                        )

                    with gr.Row():
                        enable_split = gr.Checkbox(
                            label="Split Audio by Chapter/Section",
                            value=False,
                            info="Create separate audio files for each chapter or section"
                        )

                    with gr.Row():
                        split_method = gr.Radio(
                            choices=["Auto detect", "Chapters (for EPUB)", "Sections (using pattern)", "Paragraphs"],
                            value="Auto detect",
                            label="Split Method",
                            info="How to split the input text",
                            visible=False
                        )

                    enable_split.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[enable_split],
                        outputs=[split_method]
                    )

                with gr.Accordion("Sound Effects & Silence Settings", open=True):
                    gr.Markdown("### Background Audio (WAV or MP3)")
                    with gr.Row():
                        background_audio = gr.Audio(
                            label="Background Audio Loop (optional)",
                            type="filepath",
                            sources=["upload"]
                        )
                        background_volume = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Background Volume",
                            info="Volume level for background audio"
                        )

                    gr.Markdown("---")

                    gr.Markdown("### Section Detection")
                    with gr.Row():
                        section_pattern = gr.Textbox(
                            label="Section Pattern (regex)",
                            placeholder="e.g. ^Chapter \d+",
                            info="Regular expression to identify section headers"
                        )

                    gr.Markdown("### Spacing Controls")
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            chunk_gap = gr.Slider(
                                minimum=0.0,
                                maximum=4.0,
                                step=0.1,
                                value=0.0,
                                label="Chunk Gap (seconds)",
                                info="Silence between sentences"
                            )
                        with gr.Column():
                            voice_gap = gr.Slider(
                                minimum=0.0,
                                maximum=4.0,
                                step=0.1,
                                value=0.0,
                                label="Voice Gap (seconds)",
                                info="Silence between voice changes"
                            )
                        with gr.Column():
                            section_gap = gr.Slider(
                                minimum=0.0,
                                maximum=4.0,
                                step=0.1,
                                value=0.0,
                                label="Section Gap (seconds)",
                                info="Silence between sections"
                            )
                        with gr.Column():
                            post_split_gap = gr.Slider(
                                minimum=0.0,
                                maximum=4.0,
                                step=0.1,
                                value=0.5,
                                label="Post Split Gap (seconds)",
                                info="Silence to add at the end of each split file"
                            )

                    gr.Markdown("---")

                    gr.Markdown("### Spoken Section Markers")
                    with gr.Row():
                        with gr.Column(scale=3):
                            spoken_patterns = gr.CheckboxGroup(
                                choices=[
                                    "Chapter Headers (Chapter X, Chapter One, etc.)",
                                    "Section Headers (Section X)",
                                    "Part Headers (Part X, Part One, etc.)",
                                    "Numbered Sections (1., 2., etc.)",
                                    "Book Headers (Book X, Book One, etc.)",
                                    "All Caps Headers (NEW SECTION, etc.)",
                                ],
                                value=["Chapter Headers (Chapter X, Chapter One, etc.)"],
                                label="These markers will be read aloud with silence before and after",
                                info="Select patterns that should be spoken"
                            )
                        with gr.Column(scale=1):
                            spoken_silence = gr.Slider(
                                minimum=0.0,
                                maximum=5.0,
                                step=0.1,
                                value=1.0,
                                label="Silence Duration (seconds)",
                                info="Duration of silence to insert before and after spoken markers"
                            )

                    gr.Markdown("### Silent Divider Markers")
                    with gr.Row():
                        with gr.Column(scale=3):
                            silent_patterns = gr.CheckboxGroup(
                                choices=[
                                    "Asterisk Dividers (***)",
                                    "Dash Dividers (---)",
                                    "Equal Sign Dividers (===)",
                                    "Hash Dividers (###)",
                                    "Blank Lines (multiple consecutive empty lines)",
                                    "Markdown Headers (# Title)"
                                ],
                                value=[
                                    "Asterisk Dividers (***)",
                                    "Dash Dividers (---)",
                                    "Equal Sign Dividers (===)",
                                    "Hash Dividers (###)",
                                    "Blank Lines (multiple consecutive empty lines)"
                                ],
                                label="These markers will be replaced with silence",
                                info="Select patterns that should be replaced with silence"
                            )
                        with gr.Column(scale=1):
                            silent_duration = gr.Slider(
                                minimum=0.0,
                                maximum=5.0,
                                step=0.1,
                                value=1.0,
                                label="Replacement Silence Duration (seconds)",
                                info="Duration of silence to replace these markers with"
                            )

                    gr.Markdown("### Custom Pattern")
                    with gr.Row():
                        custom_pattern = gr.Textbox(
                            label="Custom Pattern (regex)",
                            placeholder="e.g. ^Chapter \d+",
                            info="Custom regular expression to identify additional section headers"
                        )
                        custom_pattern_type = gr.Radio(
                            choices=["Spoken", "Silent"],
                            value="Spoken",
                            label="Custom Pattern Type",
                            info="Should this pattern be spoken or replaced with silence?"
                        )
                    gr.Markdown("### Text Formatting")
                    with gr.Row():
                        convert_caps = gr.Checkbox(
                            label="Convert ALL CAPS to Title Case",
                            value=True,
                            info="Convert words in ALL CAPS to Title Case for more natural reading"
                        )

            with gr.Tab("Blended Voices"):
                gr.Markdown("""
                # 🎤 Blended Voices
                Upload blended voice files (.pt) or use voices created in the Voice Lab.
                """)

                with gr.Row():
                    voice_upload = gr.File(
                        label="Upload Blended Voice (.pt file)",
                        file_types=[".pt"],
                        file_count="multiple"
                    )

                with gr.Row():
                    upload_btn = gr.Button("Upload Blended Voice", variant="primary")
                    refresh_voices_btn = gr.Button("Refresh Voice List", variant="secondary")

                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                user_voices_list = gr.Dataframe(
                    headers=["Voice Name"],
                    datatype=["str"],
                    label="Your Uploaded Blended Voices"
                )

            with gr.Tab("Voice Lab"):
                gr.Markdown("""
                # 🧪 Voice Lab
                Experiment with voice blending to create unique voices.
                """)

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            primary_language = gr.Dropdown(
                                choices=list(MAIN_LANGUAGE_CODES.keys()),
                                value="American English",
                                label="Primary Voice Language"
                            )
                            base_voice = gr.Dropdown(
                                choices=VOICES_BY_LANGUAGE["American English"],
                                value="af_nicole",
                                label="Primary Voice"
                            )
                            primary_weight = gr.Slider(
                                label="Primary Voice Weight",
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.05
                            )

                        with gr.Row():
                            secondary_language = gr.Dropdown(
                                choices=list(LANGUAGE_CODES.keys()),
                                value="American English",
                                label="Secondary Voice Language"
                            )
                            secondary_voice = gr.Dropdown(
                                choices=VOICES_BY_LANGUAGE["American English"],
                                value=None,
                                label="Secondary Voice"
                            )
                            secondary_weight = gr.Slider(
                                label="Secondary Voice Weight",
                                minimum=0.0,
                                maximum=2.0,
                                value=0.3,
                                step=0.05
                            )

                        with gr.Row():
                            tertiary_language = gr.Dropdown(
                                choices=list(LANGUAGE_CODES.keys()),
                                value="American English",
                                label="Tertiary Voice Language"
                            )
                            tertiary_voice = gr.Dropdown(
                                choices=VOICES_BY_LANGUAGE["American English"],
                                value=None,
                                label="Tertiary Voice (Optional)"
                            )
                            tertiary_weight = gr.Slider(
                                label="Tertiary Voice Weight",
                                minimum=0.0,
                                maximum=2.0,
                                value=0.1,
                                step=0.05
                            )

                        lab_text = gr.Textbox(
                            label="Sample Text",
                            value="The quick brown fox jumps over the lazy dog.",
                            lines=2
                        )

                        generate_btn = gr.Button("Generate Voice Sample", variant="primary")

                    with gr.Column():
                        lab_audio = gr.Audio(
                            label="Generated Sample",
                            type="filepath",
                            interactive=False
                        )

                        with gr.Column():
                            save_voice_btn = gr.Button("Save Blended Voice", variant="primary")
                            save_status = gr.Textbox(label="Save Status", interactive=False)

            language.change(
                fn=lambda lang: gr.Dropdown(choices=VOICES_BY_LANGUAGE.get(lang, []),
                                            value=VOICES_BY_LANGUAGE.get(lang, [])[0] if VOICES_BY_LANGUAGE.get(lang, []) else None),
                inputs=[language],
                outputs=[voice]
            )

            primary_language.change(
                fn=lambda lang: gr.Dropdown(choices=VOICES_BY_LANGUAGE.get(lang, []),
                                           value=VOICES_BY_LANGUAGE.get(lang, [])[0] if VOICES_BY_LANGUAGE.get(lang, []) else None),
                inputs=[primary_language],
                outputs=[base_voice]
            )

            secondary_language.change(
                fn=lambda lang: gr.Dropdown(choices=VOICES_BY_LANGUAGE.get(lang, []),
                                           value=VOICES_BY_LANGUAGE.get(lang, [])[0] if VOICES_BY_LANGUAGE.get(lang, []) else None),
                inputs=[secondary_language],
                outputs=[secondary_voice]
            )

            tertiary_language.change(
                fn=lambda lang: gr.Dropdown(choices=VOICES_BY_LANGUAGE.get(lang, []),
                                           value=VOICES_BY_LANGUAGE.get(lang, [])[0] if VOICES_BY_LANGUAGE.get(lang, []) else None),
                inputs=[tertiary_language],
                outputs=[tertiary_voice]
            )

            upload_btn.click(
                fn=upload_voice_file,
                inputs=[voice_upload],
                outputs=[upload_status, user_voices_list]
            )

            refresh_voices_btn.click(
                fn=refresh_user_voices,
                inputs=[],
                outputs=[user_voices_list]
            )

            iface.load(
                fn=refresh_user_voices,
                inputs=[],
                outputs=[user_voices_list]
            )

            save_voice_btn.click(
                fn=lambda primary, primary_weight, primary_lang,
                              secondary, sec_weight,
                              tertiary, tert_weight:
                      save_voice(
                          primary, primary_weight,
                          secondary, sec_weight,
                          tertiary, tert_weight,
                          LANGUAGE_CODES.get(primary_lang, "a")
                      )[0],
                inputs=[
                    base_voice, primary_weight, primary_language,
                    secondary_voice, secondary_weight,
                    tertiary_voice, tertiary_weight
                ],
                outputs=save_status
            ).then(
                fn=refresh_user_voices,
                inputs=[],
                outputs=[user_voices_list]
            )

            with gr.Tab("Help & Examples"):
                with gr.Accordion("General Help", open=True):
                    gr.Markdown(GENERAL_HELP)

            with gr.Tab("Phonemes"):
                with gr.Accordion("Phoneme Workshop", open=True):
                    with gr.Row():
                        with gr.Column():
                            phoneme_input = gr.Textbox(
                                label="Text",
                                value="This is [Kokoro](/kˈOkəɹO/), a Text-to-Speech system.",
                                lines=3
                            )

                            example_voice = gr.Dropdown(
                                choices=VOICES_BY_LANGUAGE["American English"],
                                value="af_nicole",
                                label="Voice"
                            )

                            generate_phoneme_btn = gr.Button("Generate Audio", variant="primary")

                        with gr.Column():
                            phoneme_audio = gr.Audio(
                                label="Workshop Audio",
                                type="filepath",
                                interactive=False
                            )

                gr.Markdown("""
                # 🔤 Phoneme Reference Guide

                Phonemes are the building blocks of speech sounds. This guide helps you understand and use phonemes with Kokoro TTS.
                """)

                with gr.Accordion("Using Phonemes", open=True):
                    gr.Markdown("""
                    ## How to Use Phonemes in Your Text Input

                    You can use phonemes directly in your text using Markdown link syntax:

                    ```
                    [Word](/phonemes/)
                    ```

                    For example:
                    ```
                    [Kokoro](/kˈOkəɹO/)
                    ```

                    The text before the slash is what appears in the text, and the phonemes between the slashes determine pronunciation.

                    ## Common Use Cases for Phonemes

                    1. **Unusual Names**: Make names pronounced correctly: `[Euler](/ˈɔɪlɚ/)`
                    2. **Technical Terms**: Help with technical jargon: `[SQL](/ˌɛs kju ˈɛl/)`
                    3. **Foreign Words**: Correctly pronounce words from other languages: `[Bonjour](/bɔ̃ˈʒuʁ/)`
                    4. **Emphasis Control**: Fine-tune which words get emphasized by using stress markers

                    ## Tips for Working with Phonemes

                    1. **Stress marks are important**: Use `ˈ` before stressed syllables for primary stress, `ˌ` for secondary stress
                    2. **Keep it simple**: Start with small adjustments to problem words
                    3. **Test iteratively**: Generate samples to hear how your phonemes sound
                    4. **Mix with voice tokens**: Combine with voice switching for maximum control: `[af_bella][Name](/fəˈnɛtɪk/)`
                    """)

                with gr.Accordion("What are Phonemes?", open=True):
                    gr.Markdown("""
                    **Phonemes** are the smallest units of sound in speech that can distinguish one word from another.

                    Unlike regular text, phonemes precisely define how words should be pronounced, allowing for:

                    - More accurate pronunciation control
                    - Consistent speech across different voices
                    - Custom pronunciation for unusual words or names
                    - Language-specific sounds
                    """)

                with gr.Accordion("Phoneme Dictionary", open=True):
                    gr.Markdown("""
                    ## American English Phonemes

                    | Phoneme | Example Word | Example Usage |
                    |---------|--------------|---------------|
                    | ɑ | father | f**ɑ**ðər |
                    | æ | cat | k**æ**t |
                    | ʌ | but | b**ʌ**t |
                    | ɔ | dog | d**ɔ**g |
                    | ə | about | **ə**baʊt |
                    | ɛ | red | ɹ**ɛ**d |
                    | i | see | s**i** |
                    | ɪ | sit | s**ɪ**t |
                    | u | blue | bl**u** |
                    | ʊ | book | b**ʊ**k |
                    | eɪ | say | s**eɪ** |
                    | aɪ | my | m**aɪ** |
                    | aʊ | how | h**aʊ** |
                    | ɔɪ | boy | b**ɔɪ** |
                    | p | pen | **p**ɛn |
                    | b | big | **b**ɪg |
                    | t | talk | **t**ɔk |
                    | d | dog | **d**ɔg |
                    | k | cat | **k**æt |
                    | g | go | **g**oʊ |
                    | f | fun | **f**ʌn |
                    | v | very | **v**ɛɹi |
                    | θ | think | **θ**ɪŋk |
                    | ð | this | **ð**ɪs |
                    | s | say | **s**eɪ |
                    | z | zebra | **z**ibɹə |
                    | ʃ | she | **ʃ**i |
                    | ʒ | measure | mɛ**ʒ**ər |
                    | h | hello | **h**ɛloʊ |
                    | m | me | **m**i |
                    | n | no | **n**oʊ |
                    | ŋ | sing | sɪ**ŋ** |
                    | l | light | **l**aɪt |
                    | ɹ | run | **ɹ**ʌn |
                    | j | yes | **j**ɛs |
                    | w | we | **w**i |

                    ## Stress Markers

                    | Symbol | Description | Example |
                    |--------|-------------|---------|
                    | ˈ | Primary stress | ˈæpəl (apple) |
                    | ˌ | Secondary stress | ˌʌndərˈstænd (understand) |
                    """)
            with gr.Tab("Debug Logs"):
                gr.Markdown("""
                # 🔍 Debug Logs
                View the latest log entries from the system. Click refresh to update.
                """)

                with gr.Row():
                    logs_display = gr.TextArea(
                        value="Click 'Refresh Logs' to load...",
                        label="System Logs",
                        lines=30,
                        interactive=False
                    )
                    with gr.Column():
                        refresh_btn = gr.Button("Refresh Logs", variant="secondary")
                        clear_btn = gr.Button("Clear Logs", variant="secondary")

        def toggle_input(choice):
            """Handle switching between text and file input methods and reset fields."""
            if choice == "Text Input":
                return (
                    gr.update(visible=True, value=""),
                    gr.update(visible=False, value=None)
                )
            else:
                return (
                    gr.update(visible=False, value=""),
                    gr.update(visible=True, value=None)
                )

        input_type.change(
            fn=toggle_input,
            inputs=input_type,
            outputs=[input_text, input_file]
        )

        language.change(
            fn=lambda lang: gr.Dropdown(
                choices=VOICES_BY_LANGUAGE.get(lang, []),
                value=VOICES_BY_LANGUAGE.get(lang, [])[0] if VOICES_BY_LANGUAGE.get(lang, []) else None
            ),
            inputs=[language],
            outputs=[voice]
        )

        refresh_btn.click(
            fn=load_logs,
            inputs=[],
            outputs=[logs_display]
        )

        tabs.select(
            fn=load_logs,
            inputs=[],
            outputs=[logs_display],
            show_progress=False
        )

        clear_btn.click(
            fn=clear_logs,
            inputs=[],
            outputs=[logs_display]
        ).then(
            fn=load_logs,
            inputs=[],
            outputs=[logs_display]
        )

        def handle_input_change(file, text, input_method, preview_type):
            """Handle all UI updates when input changes or preview type changes"""
            preview_text = generate_preview(file, text, input_method, preview_type)

            return (
                preview_text,
                gr.Audio(value=None, label="Generated Audio", visible=False, interactive=False),
                gr.File(value=None, label="Processing Status", visible=True, interactive=False)
            )

        input_file.change(
            fn=handle_input_change,
            inputs=[input_file, input_text, input_type, preview_type],
            outputs=[text_preview, audio_preview, output_file]
        )

        input_text.change(
            fn=handle_input_change,
            inputs=[input_file, input_text, input_type, preview_type],
            outputs=[text_preview, audio_preview, output_file]
        )

        preview_type.change(
            fn=handle_input_change,
            inputs=[input_file, input_text, input_type, preview_type],
            outputs=[text_preview, audio_preview, output_file]
        )

        generate_phoneme_btn.click(
            fn=generate_phoneme_example,
            inputs=[phoneme_input, example_voice],
            outputs=[phoneme_audio]
        )

        convert_btn.click(
            fn=lambda: {"status": "starting"},
            outputs=progress
        ).then(
            fn=process_tts,
            inputs=[
                input_file, input_text, language, voice, speed, format,
                section_pattern, spoken_patterns, silent_patterns,
                spoken_silence, silent_duration, custom_pattern, custom_pattern_type,
                chunk_gap, voice_gap, section_gap,
                background_audio, background_volume, convert_caps,
                output_dir, enable_split, split_method, post_split_gap
            ],
            outputs=[progress, output_file]
        ).then(
            fn=update_ui,
            inputs=[progress],
            outputs=[audio_preview, output_file, split_output_info]
        )

        def get_voice_sample(primary, primary_weight, primary_lang, text,
                            secondary, sec_lang, sec_weight,
                            tertiary, tert_lang, tert_weight):
            result = generate_voice_sample(
                primary, text, primary_weight,
                secondary, sec_weight,
                tertiary, tert_weight,
                LANGUAGE_CODES.get(primary_lang, "a")
            )
            return result[0] if isinstance(result, tuple) and len(result) > 0 else None

        generate_btn.click(
            fn=get_voice_sample,
            inputs=[
                base_voice, primary_weight, primary_language, lab_text,
                secondary_voice, secondary_language, secondary_weight,
                tertiary_voice, tertiary_language, tertiary_weight
            ],
            outputs=[lab_audio]
        )

    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()