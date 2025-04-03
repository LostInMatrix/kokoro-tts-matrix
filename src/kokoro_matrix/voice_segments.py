import re
from dataclasses import dataclass
from typing import List, Dict
from .constants import AUDIO_EFFECTS
import logging

logging.basicConfig(filename='gradio_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class VoiceSegment:
    """Represents a segment of text with its associated voice."""
    text: str
    voice: str
    effect_type: str = ''
    effect_volume: float = 100.0
    effect_duration: float = 0.0
    is_effect: bool = False

def parse_voice_and_effect_tokens(text: str, default_voice: str) -> List[VoiceSegment]:
    """Represents a segment of text with its associated voice."""
    segments = []
    current_voice = default_voice
    remaining_text = text
    voice_pattern = r'\[(/?[a-z]{2}_[a-z0-9_]+)\]'
    effect_pattern = r'\[effect:(\w+)(?::([0-9]*\.?[0-9]+))?\]'

    logging.debug(f"Starting to parse text: '{text}'")
    logging.debug(f"Default voice: {default_voice}")
    logging.debug(f"Voice pattern: {voice_pattern}")
    logging.debug(f"Effect pattern: {effect_pattern}")

    while remaining_text:
        voice_match = re.search(voice_pattern, remaining_text)
        effect_match = re.search(effect_pattern, remaining_text)

        logging.debug(f"Remaining text: '{remaining_text}'")
        logging.debug(f"Voice match: {voice_match}")
        logging.debug(f"Effect match: {effect_match}")

        match_pos = float('inf')
        match_type = None
        if voice_match:
            match_pos = voice_match.start()
            match_type = 'voice'
        if effect_match and effect_match.start() < match_pos:
            match_pos = effect_match.start()
            match_type = 'effect'

        logging.debug(f"Match type: {match_type}, Match position: {match_pos}")

        if match_type is None:
            if remaining_text.strip():
                logging.debug(f"Adding final text segment: '{remaining_text.strip()}'")
                segments.append(VoiceSegment(text=remaining_text.strip(), voice=current_voice))
            break

        text_before = remaining_text[:match_pos].strip()
        if text_before:
            logging.debug(f"Adding text before match: '{text_before}'")
            segments.append(VoiceSegment(text=text_before, voice=current_voice))

        if match_type == 'voice':
            token = voice_match.group(1)
            logging.debug(f"Processing voice token: {token}")
            if token.startswith('/'):
                current_voice = default_voice
                logging.debug(f"Reset voice to default: {default_voice}")
            else:
                current_voice = token
                logging.debug(f"Set current voice to: {current_voice}")
            remaining_text = remaining_text[voice_match.end():]
        else:
            effect_type = effect_match.group(1)
            value_str = effect_match.group(2)
            value = float(value_str) if value_str else 100.0

            logging.debug(f"Processing effect: {effect_type}, value: {value}")

            if effect_type == 'silence':
                logging.debug(f"Adding silence effect: duration={value}")
                segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    effect_type='silence',
                    effect_duration=value,
                    is_effect=True
                ))
            elif effect_type in AUDIO_EFFECTS:
                logging.debug(f"Adding audio effect: {effect_type}, volume={value}")
                segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    effect_type=effect_type,
                    effect_volume=value,
                    is_effect=True
                ))
            remaining_text = remaining_text[effect_match.end():]

    logging.debug(f"Final segments: {segments}")
    return [seg for seg in segments if seg.text or seg.is_effect]