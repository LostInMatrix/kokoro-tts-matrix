from importlib import resources
import os

USER_VOICES_DIR = os.path.join(os.getcwd(), "user_voices")
os.makedirs(USER_VOICES_DIR, exist_ok=True)

AUDIO_EFFECTS = {
    'silence': None,
    'chime': str(resources.files('kokoro_matrix') / 'assets' / 'effects' / 'chime.wav'),
}