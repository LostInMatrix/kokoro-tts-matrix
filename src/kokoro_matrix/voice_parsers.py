import re
from dataclasses import dataclass
from typing import List
from .voice_segments import VoiceSegment
from .constants import AUDIO_EFFECTS

def parse_podcast_style(text: str, default_voice: str = "af_bella") -> List[VoiceSegment]:
    """
    Parse text using podcast-style voice tokens and effects.

    Format:
    [voice_id] Speaker text here
    [effect:type:value] For effects
    [background:file:volume] For background audio

    Example:
    [af_bella] Welcome to the show!
    [effect:chime]
    [am_liam] Thanks for having me.
    """
    segments = []
    lines = text.split('\n')

    voice_pattern = r'^\s*\[([a-z]{1,2}_[a-z]+)\]\s*(.+)$'
    effect_pattern = r'^\s*\[effect:(\w+)(?::([0-9]*\.?[0-9]+))?\]\s*$'
    background_pattern = r'^\s*\[background:(.+?):([0-9]*\.?[0-9]+)\]\s*$'

    current_voice = default_voice

    for line in lines:
        line = line.strip()
        if not line:
            continue

        bg_match = re.match(background_pattern, line)
        if bg_match:
            file_path, volume = bg_match.groups()
            segments.append(VoiceSegment(
                text="",
                voice=current_voice,
                effect_type="background",
                effect_volume=float(volume),
                is_effect=True
            ))
            continue

        effect_match = re.match(effect_pattern, line)
        if effect_match:
            effect_type, value = effect_match.groups()
            if effect_type in AUDIO_EFFECTS or effect_type == 'silence':
                value = float(value) if value else 100.0
                segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    effect_type=effect_type,
                    effect_volume=value if effect_type != 'silence' else 100.0,
                    effect_duration=value if effect_type == 'silence' else 0.0,
                    is_effect=True
                ))
            continue

        voice_match = re.match(voice_pattern, line)
        if voice_match:
            voice, text = voice_match.groups()
            current_voice = voice
            if text.strip():
                segments.append(VoiceSegment(
                    text=text.strip(),
                    voice=current_voice
                ))
        else:
            if line.strip():
                segments.append(VoiceSegment(
                    text=line.strip(),
                    voice=current_voice
                ))

    return segments

def parse_book_style(text: str, narrator_voice: str = "af_bella") -> List[VoiceSegment]:
    """
    Parse text using book-style voice tokens and effects.

    Format:
    - Narration uses default voice
    - Dialogue wrapped in voice tags: [voice_id]"Speech"[/voice_id]
    - Effects: [effect:type:value]
    - Background: [background:file:volume]

    Example:
    The wind howled softly.
    [af_bella]"What was that?"[/af_bella] Sarah whispered.
    [effect:silence:1.0]
    [am_liam]"Just the wind."[/am_liam]
    """
    segments = []
    current_voice = narrator_voice
    remaining_text = text

    voice_pattern = r'\[([a-z]{1,2}_[a-z]+)\](.*?)\[/\1\]'
    effect_pattern = r'\[effect:(\w+)(?::([0-9]*\.?[0-9]+))?\]'
    background_pattern = r'\[background:(.+?):([0-9]*\.?[0-9]+)\]'

    while remaining_text:
        bg_match = re.match(background_pattern, remaining_text)
        if bg_match:
            file_path, volume = bg_match.groups()
            segments.append(VoiceSegment(
                text="",
                voice=current_voice,
                effect_type="background",
                effect_volume=float(volume),
                is_effect=True
            ))
            remaining_text = remaining_text[bg_match.end():].lstrip()
            continue

        effect_match = re.match(effect_pattern, remaining_text)
        if effect_match:
            effect_type, value = effect_match.groups()
            if effect_type in AUDIO_EFFECTS or effect_type == 'silence':
                value = float(value) if value else 100.0
                segments.append(VoiceSegment(
                    text="",
                    voice=current_voice,
                    effect_type=effect_type,
                    effect_volume=value if effect_type != 'silence' else 100.0,
                    effect_duration=value if effect_type == 'silence' else 0.0,
                    is_effect=True
                ))
            remaining_text = remaining_text[effect_match.end():].lstrip()
            continue

        voice_match = re.search(voice_pattern, remaining_text)
        if voice_match:
            narration = remaining_text[:voice_match.start()].strip()
            if narration:
                segments.append(VoiceSegment(
                    text=narration,
                    voice=narrator_voice
                ))

            voice, text = voice_match.groups()
            if text.strip():
                segments.append(VoiceSegment(
                    text=text.strip(),
                    voice=voice
                ))

            remaining_text = remaining_text[voice_match.end():].lstrip()
        else:
            narration = remaining_text.strip()
            if narration:
                segments.append(VoiceSegment(
                    text=narration,
                    voice=narrator_voice
                ))
            break

    return segments