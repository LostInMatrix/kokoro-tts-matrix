import re
import json
import os
import logging

DEFAULT_PRONUNCIATIONS = {
    "bow": "/baʊ/",
    "bowed": "/baʊd/",
    "breathed": "/bɹiːðd/"
}

class PronunciationDictionary:
    def __init__(self, dictionary_path=None):
        self.dictionary_path = dictionary_path or os.path.join(os.getcwd(), "pronunciation_dictionary.json")
        self.pronunciations = DEFAULT_PRONUNCIATIONS.copy()
        self.enabled = True
        self.load_dictionary()

    def load_dictionary(self):
        if os.path.exists(self.dictionary_path):
            try:
                with open(self.dictionary_path, 'r') as f:
                    data = json.load(f)
                    self.pronunciations.update(data.get('pronunciations', {}))
                    self.enabled = data.get('enabled', True)
            except Exception as e:
                logging.error(f"Error loading pronunciation dictionary: {e}")

    def save_dictionary(self):
        try:
            with open(self.dictionary_path, 'w') as f:
                json.dump({
                    'pronunciations': self.pronunciations,
                    'enabled': self.enabled
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving pronunciation dictionary: {e}")

    def apply_pronunciations(self, text):
        if not self.enabled:
            return text

        for word, phoneme in self.pronunciations.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            replacement = f"[{word}]({phoneme})"
            text = re.sub(pattern, replacement, text)

        return text

    def add_pronunciation(self, word, phoneme):
        self.pronunciations[word] = phoneme
        self.save_dictionary()

    def remove_pronunciation(self, word):
        if word in self.pronunciations:
            del self.pronunciations[word]
            self.save_dictionary()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.save_dictionary()