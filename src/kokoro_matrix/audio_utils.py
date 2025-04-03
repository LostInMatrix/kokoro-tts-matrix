import os
from pydub import AudioSegment
import subprocess
from typing import Optional

def mix_with_background(main_audio_path: str, background_audio_path: str,
                       background_volume: float = 0.3) -> str:
    """
    Mix TTS audio with a background loop at specified volume.
    """
    try:
        main_audio = AudioSegment.from_file(main_audio_path)
        background = AudioSegment.from_file(background_audio_path)
        main_length_ms = len(main_audio)
        background_length_ms = len(background)
        loops_needed = main_length_ms / background_length_ms
        looped_background = background * (int(loops_needed) + 1)
        looped_background = looped_background[:main_length_ms]
        looped_background = looped_background - (20 * (1 - background_volume))
        mixed = main_audio.overlay(looped_background)

        output_path = main_audio_path.rsplit('.', 1)[0] + '_with_background.' + main_audio_path.rsplit('.', 1)[1]
        if os.path.exists(output_path):
            os.remove(output_path)

        mixed.export(output_path, format=output_path.split('.')[-1])
        return output_path
    except Exception as e:
        print(f"Error mixing audio: {e}")
        return main_audio_path

def convert_to_mp3(wav_path: str, mp3_path: str) -> None:
    """Convert WAV file to MP3 using ffmpeg."""
    import subprocess
    try:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', wav_path,
            '-codec:a', 'libmp3lame',
            '-qscale:a', '2',
            mp3_path
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        raise