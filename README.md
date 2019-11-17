# chABSA-dataset
create dataset for NLP from chABSA-dataset 

本ファイルでは、chABSA-datasetのデータを使用して、感情分析（0：ネガティブ、1：ポジティブ）を2値クラス分類するためのデータファイル（tsv)を作成します。  

下記サイトからchABSA-dataset.zipをダウンロードして解凍します。  

https://github.com/chakki-works/chABSA-dataset  

データファイルは230個、文章データは2813個あります。  

Create_data_from_chABSA.ipynbを実行すると、以下のように「文章   感情スコア」の形式で学習データが生成されます。

```
その一方で、中国経済の景気減速や米国新政権の政策運営、英国のＥＵ離脱等のリスクにより、先行きは依然として不透明な状況にあります	0	
化粧品・雑貨事業は、大型店による店舗展開を強化し、デジタル販促による集客やイベント開催による顧客の増大に取組み、売上高は32億62百万円（前年同期比15.5％減）となりました	0	
加えて、保守契約が堅調に増加し、売上高は6,952百万円（前年同期比1.2％増）となりました	1	
利益につきましては、取替工事の増加及び保守契約による安定的な利益の確保により、セグメント利益（営業利益）は1,687百万円（前年同期比2.4％増）となりました	1	
その他のセグメントでは駐輪システムが堅調に推移し、売上高は721百万円（前年同期比0.8％増）となりました	1	
```

訓練データを70%、テストデータを30%の割合で分割しています。  

train.tsv(訓練データ7割）  
test.tsv(テストデータ3割）  

# BERTモデルの作成～学習～推論

chABSAデータセットを用いたネガティブ分類BERTモデルの作成、学習、推論については以下のファイルを参照。  

BERTモデル作成～学習~推論.ipynb

## 1.前提

OS: Ubuntu  
BERTモデル:京都大学が公開している[pytorch-pretrained-BERTモデル](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)をベースにファインチューニングを行う。   
形態素解析:Juman++ (v2.0.0-rc2) or (v2.0.0-rc3)  
ライブラリ:Pytorch  


## 2.環境構築

PyTorchでBERT日本語Pretrainedモデルを利用できる環境を構築します。

## pytorch のインストール

```python

conda create -n pytorch python=3.6
conda activate pytorch
conda install pytorch=0.4 torchvision -c pytorch
conda install pytorch=0.4 torchvision cudatoolkit -c pytorch
conda install pandas jupyter matplotlib scipy scikit-learn pillow tqdm cython
```

## Juman++のインストール

今回利用するBERT日本語Pretrainedモデルは、入力テキストにJuman++ (v2.0.0-rc2)で形態素解析を行っていますので、本記事でも形態素解析ツールを**Juman++**に合わせます。  
Juman++の導入手順は別記事でまとめていますので、以下を参照ください。

[**JUMAN++の導入手順まとめ**]https://sinyblog.com/deaplearning/juman/


## BERT日本語Pretrainedモデルの準備

BERT日本語Pretrainedモデルは以下のURLからダウンロードできます。  

[BERT日本語Pretrainedモデル]http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB

上記HPの「**Japanese_L-12_H-768_A-12_E-30_BPE.zip (1.6G)****」から**Japanese_L-12_H-768_A-12_E-30_BPE .zip**をダウンロードします。  
zipファイルを解凍するといくつかファイルが入っていますが、今回必要なものは以下の３つです。

bert_config.json:BERTモデル用のConfigファイル  
pytorch_model.bin:pytorch版BERT (pytorch-pretrained-BERT)用に変換したモデル  
vocab.txt:BERT用語録辞書データ  

## 推論とAttention可視化の実行

推論時にtorchtextで生成したTEXTオブジェクト（torchtext.data.field.Field）を利用するため、一旦TEXTオブジェクトをpklファイルにダンプしておきます。

```python
from utils.predict create_vocab_text
TEXT = create_vocab_text()
```
※以下のファイルは容量が大きいためgitリポジトリには格納されていませんので、前者については京大ＨＰからダウンロード、後者はNotebookに従って学習を行いモデルパラメータを各自保存してご利用ください。

pytorch_model.bin（pytorch-pretrained-BERT)
bert_fine_tuning_chABSA_22epoch.pth(ネガポジ学習済みパラメータファイル）


**utils\predict.py**に学習済みモデルのビルド(**build_bert_model**)と推論(**predict**)のメソッドを定義してあるので、これを利用してサンプルの文章をインプットして予測値とAttentionを可視化します。
AttentionはIPythonを使ってHTMLを可視化します。

```python
from utils.config import *
from utils.predict import predict, build_bert_model
from IPython.display import HTML, display


input_text = "以上の結果、当連結会計年度における売上高1,785百万円(前年同期比357百万円減、16.7％減)、営業損失117百万円(前年同期比174百万円減、前年同期　営業利益57百万円)、経常損失112百万円(前年同期比183百万円減、前年同期　経常利益71百万円)、親会社株主に帰属する当期純損失58百万円(前年同期比116百万円減、前年同期　親会社株主に帰属する当期純利益57百万円)となりました"
net_trained = build_bert_model()
html_output = predict(input_text, net_trained)
print("======================推論結果の表示======================")
print(input_text)
display(HTML(html_output))
```

上記コードを実行すると以下のような結果が表示されます。

<img width="800" alt="推論結果.png" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/215810/94743d03-633f-3c57-6850-cd7a364bf11e.png">
