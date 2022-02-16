# nbme-score-clinical-patient-notes
![input file image]()<br>
https://www.kaggle.com/c/petfinder-pawpularity-score/overview<br>
どんなコンペ?:<br>
開催期間:<br>
![input file image]()<br>
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
|name|url|input|output|status|comment|
|----|----|----|----|----|----|
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
<br>


#### Blogs / Qiita / etc.
|name|url|status|comment|
|----|----|----|----|
<br>


#### Official Documentation or Tutorial
|name|url|status|comment|
|----|----|----|----|
<br>

#### StackOverflow
|name|url|status|comment|
|----|----|----|----|
<br>

#### GitHub
|name|url|status|comment|
|----|----|----|----|
<br>

#### Hugging Face Platform
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle Notebooks
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle Datasets
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle Discussion
|name|url|status|comment|
|----|----|----|----|
<br>



***
## Diary

#### 2022-02-05  
コンペ参加. kagglenb000e-EDAにてデータ確認着手.
<br>
<br>
<br>

#### 2022-02-08
kagglenb000e-EDAにて関連テーブルを結合したtrainデータを外部出力しスプレッドシートで一つずつ確認する作業に着手.
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
```
# kagglenb000e-EDAより
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
当てはまりそうな手法: BIOスキームのNERタスクのpsuedo-labeling学習<br>
というのもpatient_notes.csvにはtrain.csvに現れていないpn_historyが41146個(patient_notes.csvに収載されているpn_historyの総数は42146個)もあり, これらにはannotation (教師ラベル)が付与されていない. 従って, pseuedo-labelingが有効だと思われる.<br>
タスクとしてはBIOスキームのNER. ただしBタグのentity種類数は普通にやるとfeature_textの数(=917)となるが, これはやや数が多すぎる気がする. case_numごとにタスクを独立させる場合は1 case_numごとにBタグの種類は平均して9-10程度になるので丁度良いか. これは, testデータにもcase_numは存在しており, testに未知のcase_numが現れることもないと保障されている(cf. 下記引用)ため, case_numごとにNERモデルを作る, というのは理に適っているように思える.<br>
> To help you author submission code, we include a few example instances selected from the training set. When your submitted notebook is scored, this example data will be replaced by the actual test data. The patient notes in the test set will be added to the patient_notes.csv file. **These patient notes are from the same clinical cases as the patient notes in the training set.** There are approximately 2000 patient notes in the test set.

> The patient notes in the test set will be added to the patient_notes.csv file.

scoring時にhidden testデータのpn_historyがpatient_notes.csvに追加されるというのはどういうことか. 現時点で除去されているのはリーク防止のためだとすぐに分かるが, なぜわざわざscoring時に追加する必要があるのか?<br>
これは推論時もpatient_notes.csvを学習プールとしてpseudo-labelingしつつモデルをアップデートできるようにする配慮なのだろうか?
<br>
<br>
<br>

#### 2022-05-03
結果は/だった. <br>
![input file image]()

{所感}
<br>
<br>
<br>




