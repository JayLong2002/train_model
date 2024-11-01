# 这个目录下要有label和feature
MEIJU_HOME = "/home/ubuntu/zlb/MEIJU"

# 选择特征预训练的模型(UTT)
# 文本：chinese-roberta-wwm-ext-4-UTT, bert-base-chinese-4-UTT, deberta-chinese-large-4-UTT
TEXT_FEATURE = "bert-base-chinese-4-UTT"

# 文本历史：bert-base-chinese-4-FRA
TEXT_FRAM_FEATURE = "bert-base-chinese-4-FRA"

# 语音：vggish_UTT，wav2vec-large-c-UTT
AUDIO_FEATURE = "wav2vec-large-c-UTT"

# 视觉：manet_UTT ， wav2vec-large-c-UTT
VIDEO_FEATURE = "manet_UTT"

# 生成数据的路径，目前还没有数据
GENERATOR_FEATURE = "null"

# 单独训练transformer的模态
# 一共有文本，音频，视频三种模态  ['text','audio','video']
TRAIN_DIM = 'text'

# 用于指定最好的模型放在哪里
BEST_MODEL_SAVE_DIR = "/home/ubuntu/zlb/MEIJU/model/save/"


# -------------------- don't modify -------------------------

TRAIN_TEXT_FEATURE_DIR = f"{MEIJU_HOME}/feature/mandarin_training_validation_10_30/{TEXT_FEATURE}/"

TRAIN_AUDIO_FEATURE_DIR = f"{MEIJU_HOME}/feature/mandarin_training_validation_10_30/{AUDIO_FEATURE}/"

TRAIN_VIDEO_FEATURE_DIR  = f"{MEIJU_HOME}/feature/mandarin_training_validation_10_30/{VIDEO_FEATURE}/"

TRAIN_LABEL = f"{MEIJU_HOME}/label/Training_label.csv"

VILID_LABEL = f"{MEIJU_HOME}/label/Validation_label.csv"

TEST_IDX = f"{MEIJU_HOME}/label/Testing_label.csv"

TEST_TEXT_FEATURE_DIR = f"{MEIJU_HOME}/feature/mandarin_test_10_30/{TEXT_FEATURE}/"

TEST_AUDIO_FEATURE_DIR = f"{MEIJU_HOME}/feature/mandarin_test_10_30/{AUDIO_FEATURE}/"

TEST_VIDEO_FEATURE_DIR = f"{MEIJU_HOME}/feature/mandarin_test_10_30/{VIDEO_FEATURE}/"

# 下面是关于intent，emotion为了模型训练所映射的数字mapping

intent2number = {'neutral': 4, 'questioning': 5, 'encouraging': 3, 'suggesting': 6, 'consoling': 2, 'wishing': 7, 'agreeing': 1, 'acknowledging': 0}

emotion2number = {'neutral': 4, 'disgust': 1, 'happy': 3, 'anger': 0, 'sad': 5, 'fear': 2, 'surprise': 6}

number2intent = {v: k for k, v in intent2number.items()}

number2emotion = {v: k for k, v in emotion2number.items()}

import numpy as np

# 获取选择特征的维数
def get_feature_dim():
    text_feature_home = f"{MEIJU_HOME}/feature/mandarin_training_validation_10_30/{TEXT_FEATURE}/Track2_Training_00001.mkv.npy"
    audio_feature_home = f"{MEIJU_HOME}/feature/mandarin_training_validation_10_30/{AUDIO_FEATURE}/Track2_Training_00001.mkv.npy"
    video_feature_home = f"{MEIJU_HOME}/feature/mandarin_training_validation_10_30/{VIDEO_FEATURE}/Track2_Training_00001.mkv.npy"
    text_dim = np.load(text_feature_home).shape[0]
    audio_dim = np.load(audio_feature_home).shape[0]
    video_dim = np.load(video_feature_home).shape[0]
    return text_dim,audio_dim,video_dim
