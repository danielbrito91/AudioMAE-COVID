import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score)
from datetime import datetime
import models_vit 
from dataset import AudiosetDataset
from util.misc import load_model
import json

COSWARA_PATH = 'output/coswara_finetune_2025-10-27_14-42-41/checkpoint-59.pth' 
COSWARA_EVAL_JSON = 'data/coswara_eval.json'

COUGHVID_PATH = 'output/coughvid_finetune_2025-10-28_07-53-37/checkpoint-59.pth' 
COUGHVID_EVAL_JSON = 'data/coughvid_eval.json'

MODEL_NAME = 'vit_base_patch16'
LABEL_CSV = 'data/covid_labels.csv'
NUM_CLASSES = 2
TARGET_LENGTH = 1024 
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
INTRA_REPORT_PATH = f'output/evals/intra_report_{timestamp}.json'
OUTER_REPORT_PATH = f'output/evals/outer_report_{timestamp}.json'

AUDIO_CONF = {
    'mode': 'eval', 
    'num_mel_bins': 128,
    'target_length': TARGET_LENGTH,
    'freqm': 0, # No SpecAug during eval
    'timem': 0, # No SpecAug during eval
    'mixup': 0, # No mixup during eval
    'dataset': 'audioset',
    'mean': -4.2677393, # Default AudioSet mean
    'std': 4.5689974,  # Default AudioSet std
    'noise': False,
}

def _load_model(model_path: str):
    model = models_vit.__dict__[MODEL_NAME](
        num_classes=NUM_CLASSES,
        global_pool=True,
        in_chans=1,
        img_size=(TARGET_LENGTH, AUDIO_CONF['num_mel_bins']), # <--- FIX 2
    )
    model.audio_exp = True
    model.to(DEVICE)


    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)     
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        msg = model.load_state_dict(state_dict, strict=False)
        
        print(f"Model {MODEL_NAME} loaded successfully.")
        print(f"Load message: {msg}")
        
        model.eval()
        print("Model set to eval mode.")
    else:
        print(f"Error: Model checkpoint not found at {model_path}")
        exit(1)
    return model

def load_coswara_model():
    return _load_model(COSWARA_PATH)

def load_coughvid_model():
    return _load_model(COUGHVID_PATH)

def get_data_loader_eval(dataset_json_file: str):
    dataset_eval = AudiosetDataset(
        dataset_json_file=dataset_json_file,
        audio_conf=AUDIO_CONF,
        label_csv=LABEL_CSV,
    )

    data_loader_eval = DataLoader(
        dataset_eval,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    return data_loader_eval


def run_inference(model, data_loader_eval):
    all_preds = []
    all_labels = []
    all_scores = []

    # Disable gradient calculations to save memory and speed up
    with torch.no_grad():
        for i, batch in enumerate(data_loader_eval):
            print(f"Processing batch {i+1}/{len(data_loader_eval)}", end='\r')
            samples = batch[0].to(DEVICE, non_blocking=True)
            labels_raw = batch[1]
            
            if labels_raw.dim() > 1:
                labels = torch.argmax(labels_raw, dim=1).cpu().numpy()
            else:
                labels = labels_raw.cpu().numpy()
            
            outputs = model(samples) # Get logits [B, C]
            
            preds = torch.argmax(outputs, dim=1)
            
            all_scores.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    return all_preds, all_labels, all_scores

def calculate_metrics(all_preds, all_labels, all_scores, output_dict=False):
    _cm =confusion_matrix(y_true=all_labels, y_pred=all_preds)
    _tp = _cm[0, 0]
    _fn = _cm[0, 1]
    _fp = _cm[1, 0]
    _tn = _cm[1, 1]

    report = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, [c[1] for c in all_scores]),
        'precision': precision_score(y_true=all_labels, y_pred=all_preds),
        'recall': recall_score(y_true=all_labels, y_pred=all_preds),
        'f1': f1_score(all_labels, all_preds),
        'fpr': _fp / (_fp + _tn),
        'tpr': _tp / (_tp + _fn),
        'specificity': _tn / (_tn + _fp),
    }
        
    return report


def main() -> tuple[dict, dict]:
    coswara_model = load_coswara_model()
    coughvid_model = load_coughvid_model()

    coswara_data_loader = get_data_loader_eval(COSWARA_EVAL_JSON)
    coughvid_data_loader = get_data_loader_eval(COUGHVID_EVAL_JSON)

    # Intra eval
    coswara_preds, coswara_labels, coswara_scores = run_inference(coswara_model, coswara_data_loader)
    coughvid_preds, coughvid_labels, coughvid_scores = run_inference(coughvid_model, coughvid_data_loader)

    intra_report = {
        'coswara': calculate_metrics(coswara_preds, coswara_labels, coswara_scores, output_dict=True),
        'coughvid': calculate_metrics(coughvid_preds, coughvid_labels, coughvid_scores, output_dict=True),
    }
    # Outer eval
    coswara_preds_outer, coswara_labels_outer, coswara_scores_outer = run_inference(coswara_model, coughvid_data_loader)
    coughvid_preds_outer, coughvid_labels_outer, coughvid_scores_outer = run_inference(coughvid_model, coswara_data_loader)

    outer_report = {
        'coswara': calculate_metrics(coswara_preds_outer, coswara_labels_outer, coswara_scores_outer, output_dict=True),
        'coughvid': calculate_metrics(coughvid_preds_outer, coughvid_labels_outer, coughvid_scores_outer, output_dict=True),
    }

    return intra_report, outer_report

if __name__ == "__main__":
    intra_report, outer_report = main()
    with open(INTRA_REPORT_PATH, 'w') as f:
        json.dump(intra_report, f)
    with open(OUTER_REPORT_PATH, 'w') as f:
        json.dump(outer_report, f)