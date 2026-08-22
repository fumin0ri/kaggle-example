# Predict Customer Churn — reproducible Kaggle workflow

顧客解約を予測する表形式データを題材に、EDAから提出、複数モデル比較、Nested Target Encoding、
特徴量選択、HPO、OOF Hill Climbing Ensembleまでを順に進める実例です。

このブランチは、実験をただ並べるのではなく、次の問いへ順番に答える構成になっています。

1. 入力データの契約は正しいか
2. 最小構成でOOF評価と提出を最後まで作れるか
3. ID方向の分布変化はないか
4. 同じfoldで異なるモデルを公平に比較できるか
5. OOF予測を使って過度に楽観的でないアンサンブルを作れるか
6. 目的変数を使う特徴量をリークなしで生成できるか
7. 追加特徴量は本当に同じモデルのOOFを改善したか
8. 120候補からモデルごとに有効な特徴量だけを残せるか
9. 特徴量集合を固定して3モデルを再現可能にHPOできるか
10. OOFだけでHill Climbingを行い、検証済みの提出を作れるか

## ディレクトリ構成

元プロジェクトの構成とNotebook名を維持しています。生データと生成物はGit管理外です。

```text
.
├── artifacts/                 # OOF、test予測、fold score、ensemble監査結果
├── input/                     # train.csv / test.csv（Git管理外）
├── notebooks/
│   ├── 01_basic_EDA.ipynb
│   ├── 02_first_baseline.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_diverse_baseline.ipynb
│   ├── 05_ensemble.ipynb
│   ├── 06_FE.ipynb
│   ├── 06-FE2.ipynb
│   ├── 07_feature_selection.ipynb
│   ├── 08_HPO.ipynb
│   └── 09_hill_climbing_ensemble.ipynb
├── output/                    # 固定foldとNested TE parquet（Git管理外）
├── config.py                  # target、ID、fold、seedの共通設定
├── ensemble.py                # cross-fitted OOF Hill Climbing
├── features.py                # feature schema、Nested Pair TE、fold別loader
├── hpo.py                     # Optuna探索空間とJSON I/O
├── selection.py               # OOF screening、相関除去、保守的top-k選択
├── train.py                   # 前処理、5モデル定義、共通CV runner
├── validation.py              # deterministic StratifiedKFold
└── requirements.txt
```

## セットアップ

Python 3.10以上を想定しています。

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Kaggleから取得したファイルを次の場所へ置きます。

```text
input/train.csv
input/test.csv
```

期待する主な列は次のとおりです。

- ID: `id`
- target: `Churn`（trainのみ、`Yes` / `No`）
- base predictors: targetとIDを除く19列

## 実行順序

Notebookは番号順に実行します。Jupyterをリポジトリ直下または`notebooks/`から起動しても、
同じrootを検出します。

### 01 — Basic EDA

`01_basic_EDA.ipynb`では、shape、列対応、ID重複、target分布、dtype、欠損率、ユニーク数、
各特徴量と解約率の関係を確認します。

重要なのは、グラフを作る前にassertでデータ契約を固定することです。低カーディナリティの整数列
`SeniorCitizen`は連続値ではなくカテゴリとして扱います。

### 02 — First baseline and submission

`02_first_baseline.ipynb`ではXGBoostを5-fold StratifiedKFoldで評価し、次を保存します。

```text
output/stkfolds.csv
artifacts/first_xgboost_oof.csv
artifacts/first_xgboost_test.csv
artifacts/first_xgboost_fold_scores.csv
artifacts/first_xgboost_summary.csv
submission.csv
```

欠損補完とone-hot encodingはsklearn Pipelineの中に置き、各foldの学習部分だけでfitします。
fold IDを保存することで、後続モデルのOOFを同じ行同士で比較できます。

### 03 — ID drift audit

`03_EDA.ipynb`ではIDを20個の等頻度binへ分け、解約率と契約構成の変化を確認します。

IDに予測力が見えても、IDをそのまま特徴量へ入れるとは限りません。testのIDがtrainの範囲外なら、
random CVだけが高くなる可能性があるためです。IDが時間・生成バッチ・顧客groupのどれを表すかを
先に考えます。

### 04 — Diverse baselines

`04_diverse_baseline.ipynb`では、同じbase特徴量と固定foldで5モデルを比較します。

| Model | 役割 |
| --- | --- |
| CatBoost | native categorical boosting |
| XGBoost | one-hot後のnonlinear boosting |
| ExtraTrees | randomized bagging |
| Logistic Regression | linear baseline |
| MLP | smooth nonlinear model |

モデル定義は`train.py::make_model`へ集約しているため、Notebook間でパラメータが知らないうちに
ずれることを防ぎます。各モデルのOOFとtest予測は`artifacts/`へ保存します。

元実験で記録された5-fold OOF ROC AUCは次のとおりでした。

| Model | OOF AUC |
| --- | ---: |
| XGBoost | 0.915015 |
| CatBoost | 0.914134 |
| MLP | 0.912335 |
| Logistic Regression | 0.907933 |
| ExtraTrees | 0.906760 |

### 05 — Cross-fitted ensemble

`05_ensemble.ipynb`は全モデルのID、target、fold、test順をassertしてから予測行列を作ります。

元実装は全OOFでweightを最適化し、同じOOFで評価していました。このブランチでは、各foldをhold-outし、
残り4 foldだけでweightを決めるcross-fittingへ変更しています。これにより、weight学習に使った行で
そのまま性能を評価する楽観バイアスを抑えます。

foldごとのweightが不安定、またはcross-fitted weighted AUCが単純平均以下なら、コードは自動的に
単純平均を採用します。

### 06 — Nested Pair Target Encoding

`06_FE.ipynb`では16カテゴリ列の全2列組み合わせ、合計120特徴量を生成します。

Target Encodingは目的変数を使うため、次のnested構造が必要です。

```text
outer fold k
├── outer train
│   └── inner 5-fold OOFで各train行をencode
├── outer valid
│   └── outer train全体だけでmappingをfitしてencode
└── test
    └── outer train全体だけでmappingをfitしてencode
```

保存するparquetにはIDを含め、`features.py::load_nested_pair_te`がIDで再整列します。行数、重複、
欠落ID、非有限値、列不一致があれば学習前に停止します。

元実装にあった確認用1 foldの重複生成は削除しました。TEロジックもNotebookから`features.py`へ移し、
検証・再利用しやすくしています。

### 06-FE2 — Feature ablation

`06-FE2.ipynb`ではNotebook 04と同じXGBoost、Logistic Regression、MLPへ120 TE列を追加します。
base予測とは別の`fe2_*` prefixでOOF/test予測を保存するため、元artifactを上書きしません。

元実験の結果は次のとおりです。

| Model | Base OOF AUC | + Nested Pair TE | Delta |
| --- | ---: | ---: | ---: |
| XGBoost | 0.915015 | 0.914869 | -0.000146 |
| Logistic Regression | 0.907933 | 0.908638 | **+0.000705** |
| MLP | 0.912335 | 0.911726 | -0.000609 |

特徴量追加は正しく実行されていましたが、全モデルを改善しませんでした。線形モデルだけ改善したことは、
明示的なカテゴリ交互作用が線形モデルには有効だった一方、木・MLPには120個の相関した特徴量が冗長に
なった可能性と整合します。したがって、特徴量は全モデルへ一律採用せず、モデル別に採否を決めます。

### 07 — Model-specific feature selection

`07_feature_selection.ipynb`では、120個のTE候補を二段階で絞ります。

1. selection専用fold（seed=2025）用のNested TEを生成
2. leakage-safe OOF TEの単変量AUCで候補を順位付け
3. 絶対相関0.995以上のほぼ重複した候補を除外
4. `{0, 5, 10, 20, 40, 80, all}`のtop-k集合を3モデルで比較
5. paired fold deltaの`mean - 1 standard error`が最小改善幅を超える集合だけを採用

0列を候補から外さない点が重要です。追加特徴量が安定してbaseを超えなければ、そのモデルではTEを
採用しません。結果は`artifacts/selected_te_features.json`へモデル別に保存します。

### 08 — HPO and frozen confirmation

`08_HPO.ipynb`ではNotebook 07の特徴量集合を固定して、3モデルをOptunaで最適化します。

| Model | 主な探索対象 | Trial budget |
| --- | --- | ---: |
| XGBoost | depth、learning rate、sampling、L1/L2 | 30 |
| Logistic Regression | C、class weight | 20 |
| MLP | 層構成、activation、L2、batch、learning rate | 30 |

SQLite studyへtrialを保存し、中断後も再開できます。手動baselineをtrial 0としてenqueueし、HPOが
少なくとも既存設定と比較されるようにします。探索中はtest予測を作りません。

best parameter確定後、元のseed=42 foldへ列とparameterを固定し、`final_xgboost`、
`final_logisticregression`、`final_mlp`のOOF/test予測を一度だけ保存します。tuning scoreを最終scoreと
呼ばず、confirmationとの差も記録します。

### 09 — OOF Hill Climbing Ensemble and submission

`09_hill_climbing_ensemble.ipynb`では3候補のID、target、fold、test順を検証し、単体AUCとOOF相関を
確認してからHill Climbingを行います。

各foldのweightは残り4 foldのOOFだけで探索します。現在のblendを複数のstep sizeで各候補方向へ動かし、
fit側AUCが改善する一手を反復します。validation foldのtargetは、そのfoldのweight決定に使われません。

最終候補は次の3つです。

- confirmation OOFが最高の単体モデル
- 3モデルの単純平均
- cross-fitted OOF Hill Climbing

ensembleの改善がbest singleより`5e-5`未満なら、複雑化せずbest singleを提出に使います。選択した予測は
`submission_final.csv`へ、fold別weight、探索履歴、最終OOFは`artifacts/`へ保存します。

## 改善した点

- Notebook 02で`Path`をimport前に使っていた初期化順を修正
- `Baseline.ID_COLMN`を`Baseline.ID_COLUMN`へ統一
- target、ID、fold、seedを`config.py`へ集約
- 5モデルの前処理・パラメータを`train.py`へ集約
- Nested TE実装を`features.py`へ移し、Notebookの重複コードを削減
- TE生成時の重複した1-fold計算を削除
- IDでの`one_to_one`結合と有限値チェックを追加
- FE2のOOF/test予測を`fe2_*`として保存
- base artifactの意図しない上書きを防止
- ensemble weightをcross-fittingで評価
- TE候補をOOF AUCと相関でscreeningする処理を追加
- paired fold差に基づくモデル別top-k選択を追加
- Optunaの再開可能な3モデルHPOを追加
- HPO用CVとparameter固定後のconfirmation CVを分離
- cross-fitted OOF Hill Climbingと保守的な最終候補選択を追加
- 各Notebookへ目的、リーク境界、判断基準を追記

## 再現性上の注意

- Notebookは重い学習結果を誤って表示しないよう、コード整理後の出力をクリアしています。
- Notebook 07〜09はコードと解説のみで、まだ実行していません。
- `input/`、`output/`、`artifacts/`、submissionはGit管理外です。
- まずNotebook 01と02を実行し、その後は番号順に進めてください。
- 数時間かかるMLPやNested TE生成は、少数fold・少数pairで疎通してから全実行するのが安全です。
- Public leaderboardではなく、固定foldのOOF差を実験採否の主根拠にします。
