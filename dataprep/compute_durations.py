import librosa
from tqdm import tqdm
import pandas as pd

def compute_coswara_durations(coswara: pd.DataFrame, coswara_base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    audio_types = ['breathing-deep', 'breathing-shallow', 'cough-heavy', 'cough-shallow', 
                'counting-fast', 'counting-normal', 'vowel-a', 'vowel-e', 'vowel-o']

    # Duration in seconds
    duration_dict = {
            'id': [],
            'breathing-deep': [],
            'breathing-shallow': [],
            'cough-heavy': [],
            'cough-shallow': [],
            'counting-fast': [],
            'counting-normal': [],
            'vowel-a': [],
            'vowel-e': [],
            'vowel-o': [],
        }

    errors_dict = {
        'id': [],
        'audio_type': [],
        'audio_path': [],
        'error': [],
    }

    for audio_id in tqdm(coswara['id'].unique()):
        duration_dict['id'].append(audio_id)
        for audio_type in audio_types:
            audio_path = f'{coswara_base_path}/{audio_id}/{audio_type}.wav'
            
            try:
                duration_seconds = librosa.get_duration(path=audio_path)
                duration_dict[audio_type].append(duration_seconds)
            except Exception as e:
                print(f'{audio_path} is not a valid audio file')
                duration_dict[audio_type].append(0)
                errors_dict['audio_type'].append(audio_type)
                errors_dict['audio_path'].append(audio_path)
                errors_dict['error'].append(e)
    
    return pd.DataFrame(duration_dict), pd.DataFrame(errors_dict)


def compute_coughvid_durations(coughvid: pd.DataFrame, coughvid_base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    duration_dict = {
        'id': [],
        'duration': [],
    }

    errors_dict = {
        'id': [],
        'audio_path': [],
        'error': [],
    }

    for audio_id in tqdm(coughvid['uuid'].unique()):
        duration_dict['id'].append(audio_id)
        audio_path = f'{coughvid_base_path}/{audio_id}.wav'
        
        try:
            duration_seconds = librosa.get_duration(path=audio_path)
            duration_dict['duration'].append(duration_seconds)
        except Exception as e:
            print(f'{audio_path} is not a valid audio file')
            duration_dict['duration'].append(0)
            errors_dict['id'].append(audio_id)
            errors_dict['audio_path'].append(audio_path)
            errors_dict['error'].append(e)

    return pd.DataFrame(duration_dict), pd.DataFrame(errors_dict)