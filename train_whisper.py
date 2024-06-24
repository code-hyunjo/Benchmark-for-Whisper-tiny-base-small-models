import numpy as np
import pandas as pd
import os, sys, re, math, random, time, json, pickle, gc, requests, librosa, evaluate

from tqdm import tqdm
from collections import defaultdict
import itertools as it

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AdamW, WhisperModel, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

import whisper

class CustomDataset(Dataset):
    def __init__(self, df, mata_data, feature_extractor, tokenizer, processor):
        self.df = df
        self.path = mata_data['path']
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = mata_data['max_len']
        self.device = mata_data['device']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.loc[idx, 'filename']
        audio_file_path = self.path + 'audio/' + file_name.split('-')[0] + '/' + file_name + '.wav'
        text = self.df.loc[idx, 'labeltext']

        for _ in range(5):
            try:
                audio, _ = librosa.load(audio_file_path, sr=16000)
                break
            except:
                time.sleep(0.05)
                print(audio_file_path)
                continue
        input_features = self.processor(audio, return_tensors="pt", sampling_rate=16000).input_features[0]

        tokenized = self.processor.tokenizer(text,
                                             return_tensors='pt',
                                             padding='max_length',
                                             return_attention_mask=True,
                                             max_length=self.max_len)
        labels = tokenized['input_ids'][0]
        decoder_input_ids = torch.cat([torch.tensor([self.tokenizer.eos_token_id]), labels[:-1]])

        return input_features.to(self.device), labels.to(self.device), decoder_input_ids.to(self.device)

def data_loader_gen(model_name, mata_data, train_df, valid_df, test_df, feature_extractor, tokenizer, processor):
    train_dataset = CustomDataset(train_df, mata_data, feature_extractor, tokenizer, processor)
    valid_dataset = CustomDataset(valid_df, mata_data, feature_extractor, tokenizer, processor)
    test_dataset = CustomDataset(test_df, mata_data, feature_extractor, tokenizer, processor)

    train_loader = DataLoader(train_dataset, batch_size = mata_data['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = mata_data['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = mata_data['batch_size'], shuffle=False)

    return train_loader, valid_loader, test_loader

def compute_metrics(pred_ids, label_ids, tokenizer, metric):
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return cer

def running(model, mata_data, data_loader, loss_fn, optimizer, tokenizer, metric, train_mode=True):
    if train_mode:
        model.train()
    total_loss = 0
    pred_lst = []
    target_lst = []

    pbar = tqdm(data_loader)

    for i, (input_features, labels, decoder_input_ids) in enumerate(pbar):
        if train_mode:
            optimizer.zero_grad()

        audio_features = model.encoder(input_features).to(mata_data['device'])
        outputs = model.decoder(decoder_input_ids, audio_features)

        loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        pbar.set_description('\033[1m[C_loss : {:>.5}]\033[0m'.format(round(total_loss / (i+1), 4)))

        pred = torch.argmax(outputs, dim=-1)
        pred_lst.extend(pred.cpu().numpy().tolist())
        target_lst.extend(labels.cpu().numpy().tolist())

    cer_score = compute_metrics(pred_lst, target_lst, tokenizer, metric)
    total_loss /= len(data_loader)

    return model, total_loss, cer_score


def model_training(seed, mata_data, train_df, valid_df, test_df, res_dict):
    model_name = mata_data['model_name']

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe")

    train_loader, valid_loader, test_loader = data_loader_gen(model_name, mata_data, train_df, valid_df, test_df, feature_extractor, tokenizer, processor)

    model = whisper.load_model(model_name.split('-')[1])
    model.to(mata_data['device'])

    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) # nllloss

    if mata_data['start_epoch'] != 0:
        # print(os.path.isfile(mata_data['path'] + mata_data['save_model_name'].format(seed)))
        print(mata_data['path'] + mata_data['save_model_name'] + '_v_{}.pt'.format(seed))
        checkpoint = torch.load(mata_data['path'] + mata_data['save_model_name'] + '_v_{}.pt'.format(seed))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metric = evaluate.load('cer')
    
    if mata_data['start_epoch'] != 0:
        print(res_dict['result_ver'][str(seed)])
        res = res_dict['result_ver'][str(seed)]['result']
    else:
        res = []
    print(res)
    for epoch in range(mata_data['epochs']):
        if epoch < mata_data['start_epoch']:
            continue
        print('\nEpoch: {}'.format(epoch+1))
        print('---------------------')

        model, train_loss, train_cer = running(model, mata_data, train_loader, loss_fn, optimizer, tokenizer, metric, train_mode=True)
        _, valid_loss, valid_cer = running(model, mata_data, train_loader, loss_fn, optimizer, tokenizer, metric, train_mode=False)
        
        print(f'Epoch : {epoch + 1},    t_loss : {round(train_loss, 4)},   t_cer_socre : {train_cer}')
        print(f'              v_loss : {round(valid_loss, 4)},   v_cer_socre : {valid_cer}')

        res.append([epoch, train_loss, train_cer, valid_loss, valid_cer])

        res_dict['result_ver'][str(seed)]['result'] = res

        with open(mata_data['path'] + mata_data['save_logging_file_name'], 'w') as f:
            json.dump(res_dict, f, ensure_ascii=False, indent=4)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, mata_data['path'] + mata_data['save_model_name'] + '_v_{}.pt'.format(seed))
    
    return res_dict