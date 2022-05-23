# nbme-score-clinical-patient-notes
![header](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/header.png)<br>
https://www.kaggle.com/c/nbme-score-clinical-patient-notes<br>
どんなコンペ?:<br>
開催期間:<br>
![timeline](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/timeline.png)<br>
[結果](#2022-05-03)<br>  
<br>
<br>
<br>
***
## 実験管理テーブル
https://wandb.ai/riow1983/NBME-Public?workspace=user-riow1983
|commitSHA|comment|Local CV|Public LB|
|----|----|----|----|
<br>

## Late Submissions
|commitSHA|comment|Local CV|Private LB|Public LB|
|----|----|----|----|----|
<br>


## My Assets
[notebook命名規則]  
- kagglenb001{e,t,i}-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- nb001{e,t,i}-hoge.ipynb: localで新規作成されたnotebook. 
- {e:EDA, t:train, i:inference}
- kaggle platform上で新規作成され, localで編集を加えるnotebookはファイル名kagglenbをnbに変更し, 番号は変更しない.

#### Code
作成したnotebook等の説明  
|name|url|status|comment|
|----|----|----|----|
|kagglenb000e-EDA.ipynb|[URL](https://www.kaggle.com/riow1983/kagglenb000e-eda)|Done|データ確認用notebook|
|kagglenb001t-token-classifier.ipynb|[URL]()|作成中|:hugs:transformersによるtoken-classification訓練|
<br>





***
## 参考資料
#### Snipets
```js
// Auto click for Colab
function ClickConnect(){
  console.log("Connnect Clicked - Start"); 
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
  console.log("Connnect Clicked - End"); 
};
setInterval(ClickConnect, 60000)
```
<br>

```
$ watch -n 1 "nvidia-smi"
```
<br>

```python
# PyTorch device
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
<br>

```python
# Kaggle or Colab
import sys
if 'kaggle_web_client' in sys.modules:
    # Do something
elif 'google.colab' in sys.modules:
    # Do something
```
<br>

```python
# pandas

>>> df = pd.DataFrame({"A":["a1", "a2", "a3"], "B":["b1", "b2", "b3"]})
>>> df
    A   B
0  a1  b1
1  a2  b2
2  a3  b3
>>> predictions = [[0, 1],[1,2],[2,3]]
>>> predictions
[[0, 1], [1, 2], [2, 3]]
>>> df[[0,1]] = predictions
>>> df
    A   B  0  1
0  a1  b1  0  1
1  a2  b2  1  2
2  a3  b3  2  3
```
<br>

```python
# Push to LINE

import requests

def send_line_notification(message):
    import json
    f = open("../../line.json", "r")
    json_data = json.load(f)
    line_token = json_data["kagglePush"]
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

if CFG.wandb:
    send_line_notification(f"Training of {CFG.wandbgroup} has been done. See {run.url}")
else:
    send_line_notification(f"Training of {CFG.wandbgroup} has been done.")
```
<br>

```python
# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
```
<br>

```python
>>> import sys
>>> print(sys.argv)
['demo.py', 'one', 'two', 'three']

# reference: https://docs.python.org/ja/3.8/tutorial/stdlib.html
```
<br>

```bash
# JupyterのIOPub data rate exceeded エラー回避方法
!jupyter notebook --generate-config -y
!echo 'c.NotebookApp.iopub_data_rate_limit = 10000000' >> /root/.jupyter/jupyter_notebook_config.py
```
<br>

```python
# PyTorchのバージョンを1.10.1に下げる方法 (Google Colabなのでpipでやる)
os.system('pip uninstall -y torch torchvision torchaudio')
os.system('pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html')
```
<br>


#### Papers
|name|url|status|comment|
|----|----|----|----|
|Learning to select pseudo labels: a semi-supervised method for named entity recognition|[URL](https://journal.hep.com.cn/ckcest/fitee/EN/10.1631/FITEE.1800743)|アブストのみ読了|PLによるNER訓練の論文らしいがPDFが落とせない|
|DeBERTa: Decoding-enhanced BERT with Disentangled Attention|[URL](https://arxiv.org/abs/2006.03654)|Keep|-|

<br>


#### Blogs (Medium / Qiita / Others)
|name|url|status|comment|
|----|----|----|----|
|CIFAR-10を疑似ラベル（Pseudo-Label）を使った半教師あり学習で分類する|[URL](https://qiita.com/koshian2/items/f4a458466b15bb91c7cb)|読了|PLの使い所がよく分かる. 実装がKerasなのが玉に瑕.|
|(snorkel) Snorkel — A Weak Supervision System|[URL](https://towardsdatascience.com/snorkel-a-weak-supervision-system-a8943c9b639f)|読了|NLPアノテーションツールsnorkelの紹介|
|(snorkel) Snorkel and The Dawn of Weakly Supervised Machine Learning|[URL](https://dawn.cs.stanford.edu/2017/05/08/snorkel/)|Keep|NLPアノテーションツールsnorkelの紹介|
|(Python) Python map() function|[URL](https://www.journaldev.com/22960/python-map-function)|Done|map関数の使い方|
|(typing) 実践！！Python型入門(Type Hints)|[URL](https://qiita.com/papi_tokei/items/2a309d313bc6fc5661c3)|Done|typingの使い方|
|(PyTorch) 小ネタ：Pytorch で Automatic Mixed Precision (AMP) の ON/OFF をするときの話|[URL](https://tawara.hatenablog.com/entry/2021/05/31/220936)|Done|`torch.cuda.amp`のON/OFF実装小ワザ|
|(PyTorch) [GPUを簡単に高速化・省メモリ化] NVIDIAのapex.ampがPyTorchに統合されたようです|[URL](https://qiita.com/Sosuke115/items/40265e6aaf2e414e2fea)|Done|apexってNVIDIAのAMP機能のことだったのね|
|(LINE) 【kaggle入門】処理が終わった時にLINE通知する方法|[URL](https://ikesala.com/kaggle_line/)|Done|requestsで簡単実装|
|(Bash) Bash For Loop Examples|[URL](https://www.cyberciti.biz/faq/bash-for-loop/)|Done|Bashによるfor loopの書き方|
|(Kaggle) Kaggleコード遺産|[URL](https://qiita.com/kaggle_grandmaster-arai-san/items/d59b2fb7142ec7e270a5)|Keep|Kaggleに使えそうなレガシーコード群|
|(W&B) 【入門】wandbの使い方（Google Colab+PyTorch）|[URL](https://dreamer-uma.com/wandb-pytorch-google-colab/)|Done|一見分かりやすいが, [公式のColab Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)を見たほうが良い|
|(W&B) Weights & Biases の使い方|[URL](https://note.com/npaka/n/ne1e5112a796a)|[URL](https://note.com/npaka/n/ne1e5112a796a)|Done|Dashboard, Reports, Sweeps, Artifactsがそれぞれ何をするものなのかについて日本語の解説が有難い|
|(Jupyter) Jupyter Notebook で IOPub data rate exceeded エラー|[URL](https://yoshitaku-jp.hatenablog.com/entry/2018/12/15/164849)|Done|jupyter_notebook_config.pyを編集すれば対処可能|
|(Prodigy) Supervised learning is great — it's data collection that's broken|[URL](https://explosion.ai/blog/supervised-learning-data-collection)|Keep|Prodidyを検索している途上で出会ったInesの記事. 主旨がよく分からないが捨てられず.|
|A Gentle Introduction to Self-Training and Semi-Supervised Learning|[URL](https://towardsdatascience.com/a-gentle-introduction-to-self-training-and-semi-supervised-learning-ceee73178b38)|Keep|PLの手順の基礎が確認できる|
|(PyTorch) pytorch 0.4の変更点|[URL](https://qiita.com/vintersnow/items/91545c27e2003f62ebc4)|Keep|2018年4月の大改訂. <br>特に`TensorとVariableの統合`は備忘録として|
|【ミニバッチ学習と学習率】低バッチサイズから始めよ|[URL](https://dajiro.com/entry/2020/04/15/221414#:~:text=%E3%81%9D%E3%81%AE%E3%81%9F%E3%82%81%E3%83%90%E3%83%83%E3%83%81%E3%82%B5%E3%82%A4%E3%82%BA%E3%81%AE%E4%B8%8A%E9%99%90,%E8%80%83%E6%85%AE%E3%81%99%E3%81%B9%E3%81%8D%E3%81%A7%E3%81%82%E3%82%8B%E3%80%82)|Keep|特に根拠が書いてある訳ではないが参考まで|
|(Python) クラス継承でobjectクラスを継承する理由|[URL](https://teratail.com/questions/262231)|Keep|継承クラスが無い場合はobjectと書くか, 何も書かない.|
<br>


#### Documentation (incl. Tutorial)
|name|url|status|comment|
|----|----|----|----|
|(spaCy) Training Pipelines & Models|[URL](https://spacy.io/usage/training)|Keep|spaCyによるfine-tune方法|
|(snorkel) Programmatically Build Training Data|[URL](https://www.snorkel.org/)|Keep|NLPアノテーションツールのsnorkelによるLF(ラベル関数)の多数決システムを使った自動アノテーション方式の説明.<br>PLでは無い.|
|(skweak) skweak|[URL](https://spacy.io/universe/project/skweak)|Keep|snorkelと同じくLF(ラベル関数)を使った弱学習フレームワークを提案するライブラリ.<br> spaCyと統合されているが使えるのか不明.|
|(:hugs:) DeBERTa|[URL](https://huggingface.co/docs/transformers/model_doc/deberta)|Keep|:hugs:DeBERTaの解説|
|(:hugs:) Summary of the tasks|[URL](https://huggingface.co/docs/transformers/task_summary)|Done|pipeline及びAutoModelFor{task name}によるinferenceのexample.<br>しかしAutoModel+fine-tuningのexampleは無い.|
|(:hugs:) Auto Classes|[URL](https://huggingface.co/docs/transformers/model_doc/auto#auto-classes)|Done|AutoConfig, AutoModel, AutoTokenizerがあれば他に何もいらない|
|(W&B) Launch Experiments with wandb.init|[URL](https://docs.wandb.ai/guides/track/launch)|Keep|W&Bを使った実験管理についての公式ドキュメント|
|(W&B) wandb.init|[URL](https://docs.wandb.ai/ref/python/init)|Done|wandb.initに渡せる引数一覧|
|Trobe|[URL](https://pythonrepo.com/repo/som-shahlab-trove-python-deep-learning)|Keep|Trove is a research framework for building weakly supervised (bio)medical NER classifiers without hand-labeled training data.|<br>GitHubレポジトリにチュートリアルが充実しており使いやすい印象.|
|(PyTorch) INSTALLING PREVIOUS VERSIONS OF PYTORCH|[URL](https://pytorch.org/get-started/previous-versions/)|Done|Google Colabの場合はpip管理|
|(pandas) pandas.testing.assert_frame_equal|[URL](https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html#:~:text=Check%20that%20left%20and%20right,for%20use%20in%20unit%20tests.)|Done|dfが同等かテストするメソッド|
<br>

#### BBC (StackOverflow / StackExchange / Quora / Reddit / Others)
|name|url|status|comment|
|----|----|----|----|
|Annotation tools: Prodigy, Doccano, (or others)?|[URL](https://www.reddit.com/r/LanguageTechnology/comments/fefapn/annotation_tools_prodigy_doccano_or_others/)|読了|NLPアノテーションツールの優劣について(本コンペでアノテーションツールは使わないが)|
|Difference between IOB and IOB2 format?|[URL](https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format)|読了|知っていたIOBはIOB2だった|
|(pandas) How to apply a function to two columns of Pandas dataframe|[URL](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe)|Done|pandasで２列以上に同時に関数を適用させる方法|
|(Python) What does "'\'\r" do in Python?|[URL](https://www.quora.com/What-does-r-do-in-Python)|Done|キャリッジ・リターンについて|
|(Colab) Just got Colab Pro, how can I ensure that processes remain running after closing the notebook?|[URL](https://www.reddit.com/r/GoogleColab/comments/q4s7jh/just_got_colab_pro_how_can_i_ensure_that/)|Done|Colab Pro+でbackground executionが実行されていることを確認するには|
|(Python) Get number of workers from process Pool in python multiprocessing module|[URL](https://stackoverflow.com/questions/20353956/get-number-of-workers-from-process-pool-in-python-multiprocessing-module)|Done|Google Colab Pro+のnum of workersは8だった|
|(Bash) How can I display the contents of a text file on the command line?|[URL](https://unix.stackexchange.com/questions/86321/how-can-i-display-the-contents-of-a-text-file-on-the-command-line)|Done|`less filename`でJupyter上でもファイルの中身を全て表示できる|
|(Bash) Bash: Write to File|[URL](https://linuxize.com/post/bash-write-to-file/)|Done|`echo 'this is a line' >> file.txt`で良い|
|(PyTorch) Pytorch 0.2.1をバージョンダウンする方法|[URL](https://ja.stackoverflow.com/questions/49261/pytorch-0-2-1%E3%82%92%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%83%80%E3%82%A6%E3%83%B3%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95)|Done|URLからインストールする際のリンク先等は[公式ドキュメントの旧バージョンページ](https://pytorch.org/get-started/previous-versions/)に従うこと|
<br>

#### GitHub
|name|url|status|comment|
|----|----|----|----|
|(skweak) skweak: Weak supervision for NLP|[URL](https://github.com/NorskRegnesentral/skweak)|Keep|snorkelと同じくLF(ラベル関数)を使った弱学習フレームワークを提案するライブラリ.<br> spaCyと統合されているが使えるのか不明.|
|(:hugs:) huggingface/transformers|[URL](https://github.com/huggingface/transformers/tree/master/examples/pytorch)|Keep|タスクごとのデータ構造を知りたくなったらここ|
|SentencePiece|[URL](https://github.com/google/sentencepiece)|Keep|"SentencePiece is an unsupervised text tokenizer"の一言が全て.|
|(:hugs:) debert TypeError: \_softmax_backward_data(): argument 'input_dtype' (position 4) must be torch.dtype, not Tensor #16587|[URL](https://github.com/huggingface/transformers/issues/16587)|Done|Google ColabのPyTorchのバージョンが1.10から1.11に上がったことに起因する:hugs:transformersのエラー. <br>[nbroad1881によるPRにより解消された](https://github.com/huggingface/transformers/pull/16806)ものの, <br>:hugs:transformersをinputフォルダに入れて利用しているnb001tにとっては[PyTorchのバージョンを1.10に下げる](https://ja.stackoverflow.com/questions/49261/pytorch-0-2-1%E3%82%92%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%83%80%E3%82%A6%E3%83%B3%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95)ほか方法がなかった.|
|Feedback_1st|[URL](https://github.com/antmachineintelligence/Feedback_1st)|Keep|feedback prize コンペの1位解法 github repo<br>antmachineintelligenceは杭州にあるAlibaba系Ant groupのAIチームと思われ.<br>multi-task GDBの実装およびペーパーを出している点にも注目したい.|
<br>

#### Hugging Face Platform
|name|url|status|comment|
|----|----|----|----|
<br>

#### Colab Notebook
|name|url|status|comment|
|----|----|----|----|
|(W&B) Simple_PyTorch_Integration|[URL](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)|Keep|PyTorchの訓練にW&Bを組み込む公式実装例|

#### Kaggle (Notebooks)
|name|url|status|comment|
|----|----|----|----|
|QA/NER hybrid train 🚆 [NBME]|[URL](https://www.kaggle.com/nbroad/qa-ner-hybrid-train-nbme/notebook)|Reading|:hugs:transformersによるQA/NERタスク訓練 (token classification task).<br>ただしAutoModelによるbody + リニアヘッドによるtoken classificationであり, [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification)によるものでは無い.<br>PLの言及がある. 詳細は[2022-02-15](#2022-02-15).<br>多様な言語モデルを扱えるように実装がモジュール化されておりその分可読性が犠牲になっている.|
|NBME / Deberta-base baseline [train]|[URL](https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train)|Keep|:hugs:transformersによるtoken classification task.<br>ただしAutoModelによるbody + リニアヘッドによるtoken classificationであり, [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification)によるものでは無い点が面白い.|
|NBME / Deberta-base baseline [inference]|[URL](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference/notebook)|Keep|NBME / Deberta-base baseline [train]の推論版|
|NBME / pip wheels|[URL](https://www.kaggle.com/yasufuminakama/nbme-pip-wheels)|Done|:hugs:transformersとtokenizersの特定バージョンのwhlファイル|
|YoloV5 Pseudo Labeling|[URL](https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling/notebook)|Done|PL実装の参考例の一つとして|
|feedback-nn-train|[URL](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook)|Keep|feedback prize コンペの1位解法notebookで, AWPの実装の参考例<br>その他coding全般が美しく参考になる|
<br>

#### Kaggle (Datasets)
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Discussion)
|name|url|status|comment|
|----|----|----|----|
|1st solution with code(cv:0.748 lb:0.742)|[URL](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)|Keep|feedback prize コンペの1位解法<br>NBME コンペにも多大な影響を与えていた|
<br>



***
## Diary

#### 2022-02-05  
コンペ参加. [kagglenb000e-EDA.ipynb](https://www.kaggle.com/riow1983/kagglenb000e-eda)にてデータ確認着手.
<br>
<br>
<br>

#### 2022-02-08
[kagglenb000e-EDA.ipynb](https://www.kaggle.com/riow1983/kagglenb000e-eda)にて関連テーブルを結合したtrainデータを外部出力しスプレッドシートで一つずつ確認する作業に着手.
<br>
<br>
<br>

#### 2022-02-09
annotationが空欄[]になる理由について. 大半はpn_historyにfeature_textに沿った記述が無いため空欄[]になっているようだが,<br>
中にはpn_historyにfeature_textに沿った記述があるにも関わらず, annotationが空欄[]になるケースもあるようだ.<br>
e.g.,<br>
文書ID=02425_000<br>
```
# pn_history

17 yo CC palpitation 
-Started 3 months ago, , has experienced 5-7 episodes wich last 3-4 minutes, sudden onsent, no precipitations factor, no alleviating or aggravating factor 
-Do not has any visual disturbances, sweating. has has episodes during rest and exertion. has been progressively worsened. 
-Last episode associated SOB no chet pain 
-No changes in bowel movement, no heat intolerance, no skipped meals, no fever. No mood changes, no weigth loss
-Good relationship with friends, good school grades. 
-ROS: Negative except above
PMH: None 
AllergiesNKDA-NKA
Meds: Nones
FH: Mother hx of thyroid disease, Father IM at 52 
SH: College student, live with roomates, no tobacco, drinks socially,  One use of marijuana and stopped. Sexually active 1partner last yr, +Condoms. Exercises regualrly Healthy diet
```
```
# feature_text

Family-history-of-MI-OR-Family-history-of-myocardial-infarction
```
```
# annotation

[]
```
いや, これはもしかすると"Father IM at 52"が"Father MI at 52"であればannotationは空欄にならなかったかも知れないな. (IMはMIのtypoだと思うが, 試験としてはバツだろう.)<br>
とするとやはり, 記述が無いから空欄になり, 記述があれば空欄にならない, ということでいいかも知れない.<br>
<br>
ところで各case_numの行数について<br>
```
case_num=0: 1300 rows

```

<br>
<br>
<br>

#### 2022-02-10
pn_history内の表記によっては, 複数のfeature_textに分類され得る.<br>
e.g., pn_historyn内の'1 day'という表記は, feature_textの'Duration-x-1-day'にも'1-day-duration-OR-2-days-duration'にも分類されている.<br>
以下[kagglenb000e-EDA.ipynb](https://www.kaggle.com/riow1983/kagglenb000e-eda)より<br>
```
# [DEFICIENCY_RATE, {FEATURE_TEX: np.array of ANNOTATIONs}]

       [0.08,
        {'Duration-x-1-day': array(['1 DAY', '1 day', '1 day h/o', '1 day history', '1 day of',
               '1-days hitory', 'Began yesterday', 'FOR LAST 1 DAY',
               'Onset was yesterday', 'SINCE YESTERDAY', 'STARTED A DAY AGO',
               'Started yesterday', 'X 1 DAY', 'YESTERDAY', 'Yesterday',
               'began yesreday', 'began yesterday', 'for 1 day', 'for 1d',
               'for 24 hours', 'for one day', 'for the last 1 day', 'one day',
               'one day history', 'one day of', 'onset yesterday',
               'report this morning with pain', 'since yesterday',
               'since yeterday', 'started yesterday', 'starting 1 day ago',
               'staryed yesterday', 'when he woke up yesterday',
               'woke up with yesterday', 'woke up yesterday morning with pain',
               'woke up yesterday with this pain', 'woke up yeswterday with pain',
               'x 1 DAY', 'x 1 day', 'yeasterday', 'yesterday', 'yeterday'],
              dtype='<U35')}                                                                   ],
       [0.09,
        {'1-day-duration-OR-2-days-duration': array(['1 day', '1 day ago', '1 day duration', '1 day in duration', '1d',
               '1d duration', '1day', '2 day', '2 days', '2 days duration',
               'YESTERDAY', 'one day', 'one day Hx', 'past day',
               'started 1 day ago', 'stated in the morning when she woke up',
               'two day', 'x1 day', 'yesterday'], dtype='<U38')}                                                       ],
```
<br>
当てはまりそうな手法: IOB2スキームのNERタスクのpsuedo-labeling学習<br>
というのもpatient_notes.csvにはtrain.csvに現れていないpn_historyが41146個(patient_notes.csvに収載されているpn_historyの総数は42146個)もあり, これらにはannotation (教師ラベル)が付与されていない. 従って, pseuedo-labelingが有効だと思われる.<br>
タスクとしてはIOB2スキームのNER. ただしentity種類数は普通にやるとfeature_textの数(=917)となるが, これはやや数が多すぎる気がする. case_numごとにタスクを独立させる場合は1 case_numごとにentity種類数は平均して9-10程度になるので丁度良いか. これは, testデータにもcase_numは存在しており, testに未知のcase_numが現れることもないと保障されている(cf. 下記引用)ため, case_numごとにNERモデルを作る, というのは理に適っているように思える.<br>

> To help you author submission code, we include a few example instances selected from the training set. When your submitted notebook is scored, this example data will be replaced by the actual test data. The patient notes in the test set will be added to the patient_notes.csv file. **These patient notes are from the same clinical cases as the patient notes in the training set.** There are approximately 2000 patient notes in the test set.

> The patient notes in the test set will be added to the patient_notes.csv file.

scoring時にhidden testデータのpn_historyがpatient_notes.csvに追加されるというのはどういうことか. 現時点で除去されているのはリーク防止のためだとすぐに分かるが, なぜわざわざscoring時に追加する必要があるのか?<br>
これは推論時もpatient_notes.csvを学習プールとしてpseudo-labelingしつつモデルをアップデートできるようにする配慮なのだろうか? いや, hidden testの正解ラベルも一緒に供給されない限りは推論時pseudo-labelingは機能しないだろうから, そういうことでも無いか.
<br>
<br>
<br>

#### 2022-02-15
NERタスクのPL学習の例が[公開notebook](https://www.kaggle.com/nbroad/qa-ner-hybrid-train-nbme/comments#1689948)にあった. ただしパフォーマンスが改悪したとして現在は除去された模様. これはPLの不安定さによるものである可能性が高く, うまくやれば改善すると思われる. (autherによる情報操作の可能性もある.)<br>
<br>
<br>
<br>

#### 2022-02-17
本コンペの工程を念頭に置きつつ, 現時点の自分のスキルレベル(データを見て独自解法の考案まではできるが, そこからシームレスに独自解法に沿った実装に直ちに着手できるほどの実装力/知力/意欲はまだまだ乏しいレベル)にfitしたKaggle工程表なるものを考えた:
- 最初期フェーズ(全工程の16%): 自分なりのEDA, notebook以外の媒体でデータを見る, 独自解法の考案
- 初期フェーズ(全工程の16%): 公開notebookの内, 優良なものを2,3精読 + 関連ライブラリの公式ドキュメント精読
- 中期フェーズ(全工程の33%): 独自解法に沿った実装着手
- 後期フェーズ(全工程の33%): submit, 実験, チューニング, アンサンブル

例えば３ヶ月あるコンペでは, 最初期フェーズに2W, 初期フェーズに2W, 中期フェーズに4W, 後期フェーズに4W, をそれぞれ充てる.<br>
<br>
本コンペで言うと, 実際に最初期フェーズに対応する作業に2W使用し, 独自解法の考案までには至った. 今日から2月一杯は初期フェーズとして2,3選別した公開notebookを精読 + 関連ライブラリの公式ドキュメントを精読していく. これにより独自解法の実装に必要な実装力/知力/意欲を養う. 独自解法に沿った実装に着手するのは3月に入ってからとする.<br>
<br>
<br>
<br>

#### 2022-02-22
QAタスクの場合, tokenizerのtext(first sentence)にquestionを, text_pair(second sentence)にcontextを配置する.<br>
```python
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
```
https://github.com/huggingface/transformers/blob/0187c6f0ad6c0e76c8206edeb72b94ff036df4ff/examples/pytorch/question-answering/run_qa_no_trainer.py#L397-L406
<br><br>
一方, 参照中の[NBME / Deberta-base baseline [train]](https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train)では, tokenizerのtext(first sentence)にpn_historyを, text_pair(second sentence)にfeature_textを配置している.<br>
https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=26
<br><br>
なぜか? 逆では無いのか?<br>
恐らく, first sentenceに照合文たるpn_historyを配置しても何ら差し支えなく, どちらでも良いということだと思う. 面白い.<br>
それどころか, それによりlabelにはpn_historyのみ供給すれば良く, locationがsecond sentenceの長さでoffsetされることが無いため, むしろ照合文はfirst sentenceに配置するほうが便利ですらあるのかも知れない.<br>
<br>
<br>
<br>

#### 2022-02-24
各feature_textについて, 複数のcase_numで同一feature_textが出現する頻度というのはどのくらいだろうか?<br>
もしcase_numごとでfeature_textが全く共有されていないのであれば, case_numごとのモデルを作る意義はあるかも知れないが, かなり共有されているようであればむしろcase_numで分けずに一つのモデルを作る方が理に適っていると思われる.<br>
→ [exact matchではほとんど共有されていない.](https://www.kaggle.com/riow1983/kagglenb000e-eda?scriptVersionId=88607298#How-many-feature_texts-are-shared-across-case_num?)<br>
それでも, catastrophy forgettingのこともあるので, case_numごとにモデルを作るというのは擬似ラベル学習をするならばやってみる価値はあるように思われる.<br>
<br>
ところで自分で作ったEDA notebookは[Diary](#Diary)と同じく毎日見た方が良い. そうしないと忘れる.<br>
<br>
Pseudo-labelingのアルゴリズムについて:<br>
Step 1) case_numごとにモデル(弱学習器)を作り, pn_notes(unlabeld data)のcase_numごとに弱学習器で擬似ラベルを付与.<br>
Step 2) 擬似ラベルが付与されたpn_notesをtrainに縦結合し, 擬似ラベルと通常ラベルを一緒に通常の学習(case_numごとではなく統一モデルの訓練)を開始.<br>
<br>
<br>
<br>

#### 2022-02-25
各case_numだけで訓練したモデルの当該case_numだけの評価はCVで以下の通り:<br> 
```
Score of case_num 0: 0.8497251859036535
Score of case_num 1: 0.8358820832321925
Score of case_num 2: 0.7897973324094926
Score of case_num 3: 0.8632173283033953
Score of case_num 4: 0.8268399541297892
Score of case_num 5: 0.7618812753214727
Score of case_num 6: 0.8425865038359166
Score of case_num 7: 0.7858024485438837
Score of case_num 8: 0.8301636199295346
Score of case_num 9: 0.8522291598203529
```
<br>
これに対して, 統一訓練モデルの, 各case_numごとの評価がCVでどうなっているのかについては以下の通り:<br>

```
# trained on & evaluated by case_num all
========== CV ==========
Score of case_num 0: 0.8677
Score of case_num 1: 0.8708
Score of case_num 2: 0.8197
Score of case_num 3: 0.8936
Score of case_num 4: 0.8862
Score of case_num 5: 0.7970
Score of case_num 6: 0.8873
Score of case_num 7: 0.8433
Score of case_num 8: 0.8846
Score of case_num 9: 0.8934
```
これを見ると, 総じて統一訓練モデルの方が精度が高いようだ.
<br>
[git commit SHAとwandbの連動については自動で行われる](https://docs.wandb.ai/guides/track/launch#how-can-i-save-the-git-commit-associated-with-my-run)ことが分かった. wandb.run直前のgit SHAがwandb run pageに拾われるので, パラメータをfixさせたら実験を実行する前に必ずgit pushしておくこと.<br>
<br>
<br>
<br>

#### 2022-03-21
- (1) `./src/001t_token_classifier.py`にてcase_num-allで訓練した弱学習器を使い, 
- (2)`./notebooks/nb002i-token-classifier.ipynb`でpatient_notesに擬似ラベルを付与し, 
- (3)その擬似ラベル付きpatient_notesをtrainに縦結合させてcase_num-allで(1)とは無関係に訓練したモデルを使い (なお, 1 epoch 240分ほどかかったため, fold 1以降は, 2 epochsまでとした.), 
- (4)kagglenb003i-token-classifier.ipynbで推論した結果をsubmitしたところPublic LB scoreが0.866だった.

[擬似ラベルを使わない通常訓練ではPublic LB scoreは0.861](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference/notebook)なので, 擬似ラベル訓練をすると+0.005改善するようだ. なお, CVは各fold平均して0.88-0.89と更に高精度. (対して通常訓練のCVは概ね0.85未満だった.)<br>
<br>
[現在公開されているノートブックでbest scoreは0.882](https://www.kaggle.com/code/theoviel/roberta-strikes-back/notebook)なので, これを擬似ラベル訓練に変更すれば0.887となり, 現時点で銀圏になる.<br>
<br>
<br>
<br>

#### 2022-03-29
DeBERTa Large V3を使うベストスコア公開ノートブック[Deberta-v3-large baseline - DeepShare](https://www.kaggle.com/code/guanghan/deberta-v3-large-baseline-deepshare)(LB: 0.883)を真似て./notebooks/001t-token-classifier.ipynbを改良, PL訓練では無い通常訓練でLB 0.882達成. これをベースにPL訓練版をsubmitしてみたい.<br>
<br>
<br>
<br>

#### 2022-03-30
PL学習手順整理:
- [1] (Local) ./notebooks/001t-token-classifier.ipynbにてcase-num-allで通常訓練 (モデル名: model.pth)
- [2] (Local) ./notebooks/002i-token-classifier.ipynbにてmodel.pthを使ったinference, 擬似ラベルデータ作成
- [3] (Local) ./notebooks/001t-train-classifier.ipynbにてcase-num-allでPL訓練 (モデル名: model_pl.pth)
- [4] (Local) ./notebooks/001t-train-classifier/をKaggle Datasetにアップロード
- [5] (Kaggle Platform) kagglenb003i-token-classifier.ipynbにてmodel_pl.pthを使ったinference, submission.csv作成, submit

<br>
<br>
<br>

#### 2022-04-09
予告通り, LB: 0.882を達成したdeberta-v3-largeのPL訓練版をsubmitした結果, LB: 0.885となり, 銅圏最後尾(97位)につけた. CVは0.9を超えていたので, trust CVならshake upを望めるが, どうするか.<br>
<br>
<br>
<br>

#### 2022-04-10
永らくPLに関する情報はコンペ内で大々的に共有されていなかったが, ここに来てディスカッションなどでPLの利点・難点について議論されるようになってきた模様.<br>
こちらのディスカッション [Pseudo labels make cv/lb uncorrelated?](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315321) では, PLを普通にやるとCVがLBに比して優秀(>0.9)なのはPLによるリークが起きているに違いないという論点が共有されており, 過去コンペでそのようなリークを防ぐため, PLデータをfoldごとに作っておき, foldごとにconcatenateして訓練する方法が提案されていた. [1st place solution with code](https://www.kaggle.com/c/google-quest-challenge/discussion/129840)<br>
> We could expect a lot of very similar questions/answers in SO&SE data, so we clearly couldn’t rely on that sort of pseudolabels. In order to fix the problem, **we generated 5 different sets of pseudolabels where for each train/val split we used only those models that were trained using only the current train set.** This gave much less accurate pseudolabels, but eliminated any possible source of target leakage.

従って私もこの方法に従って実装を変更することにした.<br>
なお, 図にすると以下のような流れになると思われ:<br>
![pl_fold](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/pl_fold.png)
<br>
<br>
<br>

#### 2022-04-27
ようやくリーク防止型のPL訓練が5 fold分完了した. なお, 4月26日に最後の5fold目を訓練する際, debertに起因するエラー `TypeError: \_softmax_backward_data(): argument 'input_dtype' (position 4) must be torch.dtype, not Tensor` に遭遇. 4fold目まではこんなことはなかったのに変だと思っていたら, どうやらGoogle ColabのPyTorchのバージョンが1.10から1.11に上がったことで, :hugs:transformers側のバージョンチェッカーが反応していたらしい. その不具合は[こちら](https://github.com/huggingface/transformers/issues/16587)で報告され, PRがなされfixした. しかし我がnb001tは訳あって:hugs:transformersをinputフォルダからインストールしているので, fixは反映されない. PyTorchのバージョンを1.10に下げることでエラー解消でき無事5fold目の訓練も完了した.<br>
submitした結果, LB=0.887となり, リーク防止前の0.885と比べて0.002改善した.
<br>
<br>
<br>


#### 2022-05-03
結果は277/1471だった. <br>
![private lb image](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/result.png)
<br>
<br>
**どのように取り組み, 何を反省しているか**<br>
今回のコンペはコンペ開始直後から参加することができ, 与えられた期間は３ヶ月だった. そのため期間を４フェーズに分け, 最初期フェーズではEDAをしつつ独自解法の考案, 次の初期フェーズでは公開notebookのうち気に入ったものを2,3精読する, 中期フェーズでは独自解放の実装, 後期フェーズで実験, と計画的に取り組めたが, 最後の実験フェーズで計算リソースの限界からか後述する気力減退に陥ってしまった. また考案した独自解法は1個では不十分で, 10種類くらい着想していく必要があり, 全て実装して結果を確認するだけのスピードも求められると感じた.<br>
<br>
**`tokenizer(text, feature_text)`について**<br>
textにpn_history (受験者が擬似患者に問診し記述したテキスト), feature_text に当該擬似患者の真の特性 (病歴情報) が入る. 一方, labelはspan ((start_position, end_position)) であり, text[span]がfeature_textに意味的に合致すれば正解となる.<br>
Datasetを介して, textとfeature_textをinputs で受け, labelはそのままlabelで受け, modelに入力する. modelのoutputはtokenごとのlast_hidden_stateを元にした確率値のような値である. この値をlabelと同一の構造((start_position, end_position))に変換する処理はモデルの外で行い, lossは変換後の予測labelと真のlabelをBCEで計算し, 誤差逆伝播でtokenごとのlast_hidden_stateが最適化されるように仕向けている.<br>
<br>
**modelのoutputから予測label獲得までの処理について*<br>
```
# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):

    ... (省略) ...

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output
```
last_hidden_states: (n_rows, seq_len, hidden_size)<br>
`self.fc`はhidden_size -> 1 とする線形変換なので, hidden_sizeは1となる.<br>
次に, 0-1間に収めて確率のように扱えるよう, `output.sigmoid()`をvalid_fn (or inference_fn) の中で行い,<br> 
次に, valid_fn (or inference_fn) の外側 (= train loop内) で `output.reshape((n_rows, seq_len))`とする.<br>
![reshape](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/reshape.png)
<br>
これにより, 1インスタンス (1 text) ごとにtokenごとの"feature_text該当部分たり得る確率"が得られ, これをget_char_probs -> get_results の順に処理していくことで, 予測spanに変換している.<br>
![get functions](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/get_functions.png)
<br>
<br>
**リーク防止PL学習について**<br>
My submissions:<br>
![my submissions](https://github.com/riow1983/nbme-score-clinical-patient-notes/blob/main/png/mysubmissions.png)
<br>
自分で発想し実装したリーク防止PLの結果も, Private LBが通常のPLより悪くなっており, Public LBにオーバーフィットしていただけのようだったのは残念. <br>
また大幅なshake down(138位下落)となったのも痛かった.<br>
計算リソースはGoogle Colab Pro+を使用したものの, PL学習では1 foldにかかる時間が12時間と長く, 試行錯誤を繰り返す気力が萎えてしまった. これ以上の課金をしないという前提での解決策としてはPL学習に使う未ラベルデータをランダムサンプリングして減らすか, Deep Speedなどのメモリ効率化系ライブラリの実装に取り組むか, 辺りだと思う. 
<br>
<br>
<br>
Back to [Top](#nbme-score-clinical-patient-notes)



