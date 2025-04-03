import tempfile
import soundfile as sf
import logging
import numpy as np
import os
from kokoro_matrix.pipeline import initialize_pipeline
from typing import Optional, List, Dict, Any, Union

def add_voice_variation(embedding, strength: float = 0.26, seed: Optional[int] = None):
    """
    Add controlled variation to a voice embedding based on a fixed seed.

    Args:
        embedding: The voice embedding to modify
        strength: The amount of variation to add (0.0-1.0)
        seed: Random seed for reproducible results

    Returns:
        Modified voice embedding with added variation
    """
    if seed is not None:
        np.random.seed(seed)

    feature_noise = np.random.randn(embedding.shape[-1]) * strength

    varied_embedding = embedding.copy()
    for i in range(embedding.shape[0]):
        varied_embedding[i, :] += feature_noise

    return varied_embedding

def pca_perturbation(embedding, strength=0.1, n_components=5, seed: Optional[int] = None):
    """
    Apply PCA-based perturbation to a voice embedding.
    This creates more natural voice variations by only modifying
    along the principal dimensions of voice variation.
    
    Args:
        embedding: The voice embedding to modify
        strength: The amount of perturbation to apply
        n_components: Number of principal components to modify
        seed: Random seed for reproducible results
        
    Returns:
        Modified voice embedding
    """
    if seed is not None:
        np.random.seed(seed)
    
    if embedding.shape[1] == 1:
        embedding_squeezed = embedding[:, 0, :]
    else:
        embedding_squeezed = embedding
    
    flat_embedding = embedding_squeezed.T
    n_samples, n_features = flat_embedding.shape
    n_components = min(n_components, n_samples)
    
    mean = np.mean(flat_embedding, axis=1)
    centered = flat_embedding - mean[:, np.newaxis]
    cov = np.cov(centered, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    top_components = eigenvectors[:, :n_components]
    perturbation = np.random.randn(n_components) * strength
    perturbation_vector = np.dot(top_components, perturbation)
    
    perturbed_embedding = embedding.copy()
    if embedding.ndim == 3:
        perturbed_embedding += perturbation_vector[np.newaxis, np.newaxis, :]
    else:
        perturbed_embedding += perturbation_vector[np.newaxis, :]
    
    return perturbed_embedding

def spectral_envelope_modification(base_voice, formant_shift=0.1, brightness=0.8):
    """
    Modify the spectral envelope of a voice.
    
    Args:
        base_voice: The voice embedding to modify
        formant_shift: Amount to shift formants (-0.5 to 0.5)
        brightness: Brightness adjustment (0.0 to 2.0)
        
    Returns:
        Modified voice embedding
    """
    result = base_voice.copy()
    feature_dim = base_voice.shape[-1]
    shift_amount = int(feature_dim * formant_shift)
    brightness_slope = np.linspace(0, brightness, feature_dim)
    
    for i in range(base_voice.shape[0]):
        features = result[i, :]
        if shift_amount > 0:
            shifted = np.concatenate([features[-shift_amount:], features[:-shift_amount]])
        else:
            shift = abs(shift_amount)
            shifted = np.concatenate([features[shift:], features[:shift]])
        
        features = features * 0.3 + shifted * 0.7
        features *= (1.0 + brightness_slope)
        result[i, :] = features
    
    return result

def blend_voices(voice_list):
    """
    Blend multiple voices with weights.
    
    Args:
        voice_list: List of (voice_embedding, weight) tuples
        
    Returns:
        Blended voice embedding
    """
    if len(voice_list) == 1 and len(voice_list[0]) == 1:
        return voice_list[0][0]
    
    voices = [item[0] for item in voice_list]
    weights = [item[1] if len(item) > 1 else 1.0 for item in voice_list]
    total_weight = sum(weights)
    
    if total_weight == 0:
        weights = [1.0 / len(voices) for _ in voices]
    else:
        weights = [w / total_weight for w in weights]
    
    blended = np.zeros_like(voices[0])
    for voice, weight in zip(voices, weights):
        blended += voice * weight
    
    return blended
def generate_voice_sample(primary_voice, text, primary_weight=1.0,
                         secondary_voice=None, secondary_weight=0.3,
                         tertiary_voice=None, tertiary_weight=0.1, lang_code="a"):
    """Generate a voice sample with voice blending."""
    try:
        pipeline = initialize_pipeline(lang_code=lang_code)

        pipeline.load_voice(primary_voice)
        if secondary_voice:
            pipeline.load_voice(secondary_voice)
        if tertiary_voice:
            pipeline.load_voice(tertiary_voice)

        voice_list = [(pipeline.voices[primary_voice], primary_weight)]

        if secondary_voice and secondary_voice in pipeline.voices:
            voice_list.append((pipeline.voices[secondary_voice], secondary_weight))

        if tertiary_voice and tertiary_voice in pipeline.voices:
            voice_list.append((pipeline.voices[tertiary_voice], tertiary_weight))

        import torch
        if len(voice_list) > 1:
            voice_list_numpy = []
            for voice_emb, weight in voice_list:
                if isinstance(voice_emb, torch.Tensor):
                    voice_list_numpy.append((voice_emb.cpu().detach().numpy(), weight))
                else:
                    voice_list_numpy.append((voice_emb, weight))

            blended_emb = blend_voices(voice_list_numpy)

            blended_tensor = torch.tensor(blended_emb,
                                         dtype=voice_list[0][0].dtype,
                                         device=voice_list[0][0].device)

            temp_voice_name = f"{primary_voice}_blend"
            pipeline.voices[temp_voice_name] = blended_tensor
            use_voice = temp_voice_name
        else:
            use_voice = primary_voice

        audio_segments = []
        for _, _, audio in pipeline(text, voice=use_voice):
            if audio is not None:
                audio_segments.append(audio.numpy())

        if not audio_segments:
            return None, "Failed to generate audio segments"

        combined_audio = np.concatenate(audio_segments)

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, combined_audio, 24000)

        return temp_file.name, None
    except Exception as e:
        logging.exception(f"Error in generate_voice_sample: {e}")
        return None, f"Error generating sample: {str(e)}"

def save_voice(primary_voice, primary_weight=1.0,
              secondary_voice=None, secondary_weight=0.3,
              tertiary_voice=None, tertiary_weight=0.1,
              lang_code="a"):
    """
    Save a blended voice directly to USER_VOICES_DIR.

    Returns:
        tuple: (message, voice_name)
    """
    try:
        lang_prefix = primary_voice[0]
        voice_name_parts = [f"{primary_voice}_{int(primary_weight*100)}"]

        if secondary_voice:
            voice_name_parts.append(f"{secondary_voice}_{int(secondary_weight*100)}")

        if tertiary_voice:
            voice_name_parts.append(f"{tertiary_voice}_{int(tertiary_weight*100)}")

        blend_name = f"{lang_prefix}_" + "_blend_".join(voice_name_parts)

        pipeline = initialize_pipeline(lang_code=lang_code)

        pipeline.load_voice(primary_voice)
        if secondary_voice:
            pipeline.load_voice(secondary_voice)
        if tertiary_voice:
            pipeline.load_voice(tertiary_voice)

        voice_list = [(pipeline.voices[primary_voice], primary_weight)]

        if secondary_voice and secondary_voice in pipeline.voices:
            voice_list.append((pipeline.voices[secondary_voice], secondary_weight))

        if tertiary_voice and tertiary_voice in pipeline.voices:
            voice_list.append((pipeline.voices[tertiary_voice], tertiary_weight))

        import torch
        from kokoro_matrix.constants import USER_VOICES_DIR

        if len(voice_list) > 1:
            voice_list_numpy = []
            for voice_emb, weight in voice_list:
                if isinstance(voice_emb, torch.Tensor):
                    voice_list_numpy.append((voice_emb.cpu().detach().numpy(), weight))
                else:
                    voice_list_numpy.append((voice_emb, weight))

            blended_emb = blend_voices(voice_list_numpy)

            blended_tensor = torch.tensor(blended_emb,
                                         dtype=voice_list[0][0].dtype,
                                         device=voice_list[0][0].device)
        else:
            blended_tensor = voice_list[0][0]

        os.makedirs(USER_VOICES_DIR, exist_ok=True)
        voice_path = os.path.join(USER_VOICES_DIR, f"{blend_name}.pt")
        torch.save(blended_tensor, voice_path)
        pipeline.voices[blend_name] = blended_tensor

        return f"Voice saved as {blend_name}", blend_name

    except Exception as e:
        logging.exception(f"Error saving voice: {e}")
        return f"Error saving voice: {str(e)}", None