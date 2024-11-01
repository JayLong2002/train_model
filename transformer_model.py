import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

from config import TRAIN_TEXT_FEATURE_DIR,TRAIN_AUDIO_FEATURE_DIR,TRAIN_VIDEO_FEATURE_DIR,TRAIN_LABEL,VILID_LABEL,TEST_IDX

from config import TEST_TEXT_FEATURE_DIR, TEST_AUDIO_FEATURE_DIR, TEST_VIDEO_FEATURE_DIR 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalDataset(Dataset):
    def __init__(self, mode='train'):
        self.text_feature_dir = TRAIN_TEXT_FEATURE_DIR
        self.audio_feature_dir = TRAIN_AUDIO_FEATURE_DIR
        self.video_feature_dir = TRAIN_VIDEO_FEATURE_DIR

        self.mode = mode  # 标记是训练集还是验证集
        if mode == 'train':
            self.labels = pd.read_csv(TRAIN_LABEL)
        elif mode == 'valid':
            self.labels = pd.read_csv(VILID_LABEL)
        else:
            raise ValueError("Invalid mode specified. Use 'train' or 'valid'.")

        self.data = self._pre_load_data()

    def __len__(self):
        return len(self.labels)

    def _pre_load_data(self):
        data = {}
        for index, row in self.labels.iterrows():
            filename = row['FileName']
            
            feature_file_name = filename + '.npy'

            text_path = os.path.join(self.text_feature_dir, feature_file_name)
            audio_path = os.path.join(self.audio_feature_dir, feature_file_name)
            video_path = os.path.join(self.video_feature_dir, feature_file_name)
            
            text_vec = np.load(text_path)
            audio_vec = np.load(audio_path)
            video_vec = np.load(video_path)
            
            data[filename] = (text_vec, audio_vec, video_vec)
        return data
    
    def __getitem__(self, idx):
        # 从 label 文件中 读取filename 和 intent，emotion
        filename = self.labels.iloc[idx, 1]
        intent_label = self.labels.iloc[idx, 3]
        emotion_label = self.labels.iloc[idx, 2]

        text, audio, video = self.data[filename]

        # 转换为 PyTorch 张量
        text = torch.tensor(text, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)
        
        # 转换标签
        intent_label = torch.tensor(intent_label, dtype=torch.long)
        emotion_label = torch.tensor(emotion_label, dtype=torch.long)
        
        return text, audio, video, intent_label, emotion_label

class TestDataset(Dataset):
    def __init__(self):
        self.text_feature_dir = TEST_TEXT_FEATURE_DIR
        self.audio_feature_dir = TEST_AUDIO_FEATURE_DIR
        self.video_feature_dir = TEST_VIDEO_FEATURE_DIR 

        self.labels = pd.read_csv(TEST_IDX)

        self.data = self._pre_load_data()

    def __len__(self):
        return len(self.labels)

    def _pre_load_data(self):
        data = {}
        for index, row in self.labels.iterrows():
            filename = row['FileName']
            
            feature_file_name = filename + '.npy'
            text_path = os.path.join(self.text_feature_dir, feature_file_name)
            audio_path = os.path.join(self.audio_feature_dir, feature_file_name)
            video_path = os.path.join(self.video_feature_dir, feature_file_name)
            
            text_vec = np.load(text_path)
            audio_vec = np.load(audio_path)
            video_vec = np.load(video_path)
            
            data[filename] = (text_vec, audio_vec, video_vec)
        return data
    
    def __getitem__(self, idx):
        # 从 label 文件中 读取filename 和 intent，emotion
        filename = self.labels.iloc[idx, 1]

        text, audio, video = self.data[filename]

        # 转换为 PyTorch 张量
        text = torch.tensor(text, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)
        
        return text, audio, video, filename

"""
多模态模型
"""
class MultimodalEmotionRecognition(nn.Module):
    def __init__(
                    self, 
                    text_dim, audio_dim, video_dim,  # 输入维度
                    intent_classes, emotion_classes,  # 类别数
                    text_head=4, text_layers=2,  # 文本 Transformer 参数
                    audio_head=4, audio_layers=2,  # 音频 Transformer 参数
                    video_head=4, video_layers=2,  # 视频 Transformer 参数
                    drop_out=0.1, # Dropout 概率
                    text_model_path=None, # text_transformer 模型文件
                    audio_model_path=None, # audio_transformer 模型文件
                    video_model_path=None  # video_transformer 模型文件
                ):
        super(MultimodalEmotionRecognition, self).__init__()
        
        # Text, Audio, Video Transformer Encoders
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=text_dim, nhead=text_head), num_layers=text_layers
        )
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=audio_dim, nhead=audio_head), num_layers=audio_layers
        )
        self.video_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=video_dim, nhead=video_head), num_layers=video_layers
        )
        
        # 如果传入了三个transformer的模型文件，就从模型文件加载
        if text_model_path != None:
            self.text_transformer.load_state_dict(torch.load(text_model_path))

        if audio_model_path != None:
            self.audio_transformer.load_state_dict(torch.load(audio_model_path))

        if video_model_path != None:
            self.video_transformer.load_state_dict(torch.load(video_model_path))


        # 融合的维度
        combined_dim = text_dim + audio_dim + video_dim
        self.fc1 = nn.Linear(combined_dim, 256)

        self.dropout = nn.Dropout(drop_out)

        self.batch_norm = nn.BatchNorm1d(256)
        
        # Intent and Emotion Classifiers
        self.intent_classifier = nn.Linear(256, intent_classes)
        self.emotion_classifier = nn.Linear(256, emotion_classes)

    def forward(self, text, audio, video):
        # 添加额外维度用于Transformer输入 (sequence length, batch size, feature size)
        text_out = self.text_transformer(text.unsqueeze(1)).squeeze(1)
        audio_out = self.audio_transformer(audio.unsqueeze(1)).squeeze(1)
        video_out = self.video_transformer(video.unsqueeze(1)).squeeze(1)
        
        text_out = self.dropout(text_out)
        audio_out = self.dropout(audio_out)
        video_out = self.dropout(video_out)

        # 特征融合
        # combined = torch.cat((text_out, audio_out, video_out), dim=1)
        combined = torch.cat((torch.relu(text_out), torch.relu(audio_out), torch.relu(video_out)), dim=1)
        combined = self.fc1(combined)
        combined = self.batch_norm(combined)
        combined = self.dropout(torch.relu(combined))
        
        # 意图和情绪分类
        intent_output = self.intent_classifier(combined)
        emotion_output = self.emotion_classifier(combined)
        
        return intent_output, emotion_output

"""
单独训练单独模态的transformer
"""
class SingleModalEmotionRecognition(nn.Module):
    def __init__(
                        self, 
                        feature_dim,  # 输入维度
                        intent_classes, emotion_classes,  # 类别数
                        head=4,  # 头数
                        layers=8, # 维数
                        drop_out=0.1 # Dropout 概率
                ):
        super(SingleModalEmotionRecognition, self).__init__()
        
        # Text, Audio, Video Transformer Encoders
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=head), num_layers=layers
        )
    
        combined_dim = feature_dim

        self.fc1 = nn.Linear(combined_dim, 256)

        self.dropout = nn.Dropout(drop_out)

        self.batch_norm = nn.BatchNorm1d(256)
        
        # Intent and Emotion Classifiers
        self.intent_classifier = nn.Linear(256, intent_classes)
        self.emotion_classifier = nn.Linear(256, emotion_classes)

    def forward(self, feature):
        # 添加额外维度用于Transformer输入 (sequence length, batch size, feature size)
        feature_out = self.transformer(feature.unsqueeze(1)).squeeze(1)

        feature_out = self.dropout(feature_out)

        combined = feature_out
        # 特征融合
        combined = self.fc1(combined)
        combined = self.batch_norm(combined)
        combined = self.dropout(torch.relu(combined))
        
        # 意图和情绪分类
        intent_output = self.intent_classifier(combined)
        emotion_output = self.emotion_classifier(combined)
        
        return intent_output, emotion_output


