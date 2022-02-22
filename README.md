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
- localnb001{e,t,i}-hoge.ipynb: localで新規作成されたnotebook. 
- {e:EDA, t:train, i:inference}

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
<br>

#### BBC (StackOverflow / StackExchange / Quora / Reddit / Others)
|name|url|status|comment|
|----|----|----|----|
|Annotation tools: Prodigy, Doccano, (or others)?|[URL](https://www.reddit.com/r/LanguageTechnology/comments/fefapn/annotation_tools_prodigy_doccano_or_others/)|読了|NLPアノテーションツールの優劣について(本コンペでアノテーションツールは使わないが)|
|Difference between IOB and IOB2 format?|[URL](https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format)|読了|知っていたIOBはIOB2だった|
|(pandas) How to apply a function to two columns of Pandas dataframe|[URL](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe)|Done|pandasで２列以上に同時に関数を適用させる方法|
|(Python) What does "'\'\r" do in Python?|[URL](https://www.quora.com/What-does-r-do-in-Python)|Done|キャリッジ・リターンについて|
<br>

#### GitHub
|name|url|status|comment|
|----|----|----|----|
|(skweak) skweak: Weak supervision for NLP|[URL](https://github.com/NorskRegnesentral/skweak)|Keep|snorkelと同じくLF(ラベル関数)を使った弱学習フレームワークを提案するライブラリ.<br> spaCyと統合されているが使えるのか不明.|
|(:hugs:) huggingface/transformers|[URL](https://github.com/huggingface/transformers/tree/master/examples/pytorch)|Keep|タスクごとのデータ構造を知りたくなったらここ|

<br>

#### Hugging Face Platform
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Notebooks)
|name|url|status|comment|
|----|----|----|----|
|QA/NER hybrid train 🚆 [NBME]|[URL](https://www.kaggle.com/nbroad/qa-ner-hybrid-train-nbme/notebook)|Reading|:hugs:transformersによるQA/NERタスク訓練 (token classification task).<br>ただしAutoModelによるbody + リニアヘッドによるtoken classificationであり, [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification)によるものでは無い.<br>PLの言及がある. 詳細は[2022-02-15](#2022-02-15).<br>多様な言語モデルを扱えるように実装がモジュール化されておりその分可読性が犠牲になっている.|
|NBME / Deberta-base baseline [train]|[URL](https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train)|Keep|:hugs:transformersによるtoken classification task.<br>ただしAutoModelによるbody + リニアヘッドによるtoken classificationであり, [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification)によるものでは無い点が面白い.|
<br>

#### Kaggle (Datasets)
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Discussion)
|name|url|status|comment|
|----|----|----|----|
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
<iframe src="https://www.kaggle.com/embed/yasufuminakama/nbme-deberta-base-baseline-train?cellIds=26&kernelSessionId=87264998" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="NBME / Deberta-base baseline [train]"></iframe>
<br>
https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=26
<br><br>
なぜか? 逆では無いのか?

#### 2022-05-03
結果は/だった. <br>
![input file image]()

{所感}
<br>
<br>
<br>




