#  自然言語処理による感情分析（BERT）

import glob
import os
import io
import string
import re
import sys
import random
import spacy
import torchtext
import mojimoji
import string
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext
import pickle

from torchtext.vocab import Vectors
from appv1.bert import BertTokenizer, load_vocab
from appv1.config import PKL_FILE, VOCAB_FILE, DATA_PATH


def get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=32):
    """IMDbのDataLoaderとTEXTオブジェクトを取得する。 """
    # 乱数のシードを設定
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    # 単語分割用のTokenizerを用意
    tokenizer_bert = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

    def preprocessing_text(text):
        # 半角・全角の統一
        text = mojimoji.han_to_zen(text) 
        # 改行、半角スペース、全角スペースを削除
        text = re.sub('\r', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('　', '', text)
        text = re.sub(' ', '', text)
        # 数字文字の一律「0」化
        text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")
            return text

    # 前処理と単語分割をまとめた関数を定義
    # 単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
    def tokenizer_with_preprocessing(text, tokenizer=tokenizer_bert.tokenize):
        text = preprocessing_text(text)
        ret = tokenizer(text)  # tokenizer_bert
        return ret
    # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
    max_length = 256
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # フォルダ「data」から各tsvファイルを読み込みます
    # BERT用で処理するので、10分弱時間がかかります
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=DATA_PATH, train='train.tsv',
        test='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    vocab_bert, ids_to_tokens_bert = load_vocab(vocab_file=VOCAB_FILE)
    TEXT.build_vocab(train_val_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert    
    
    batch_size = 32  # BERTでは16、32あたりを使用する
    train_dl = torchtext.data.Iterator(train_val_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)
    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dl, "val": val_dl}

    return train_dl, val_dl, TEXT, dataloaders_dict






