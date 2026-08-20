# Kaggle の一連の進め方 — Spaceship Titanic 実例

このリポジトリは、[`kaggle-template`](https://github.com/fumin0ri/kaggle-template) を出発点に、
tabular competition を次の順で進める実行済みの例です。

> Simple EDA → first submission → CV audit → diverse baselines → feature engineering →
> feature selection → HPO → confirmation CV → OOF Hill Climbing Ensemble

## このリポジトリでいう「再現性」

再現性は「Spaceship Titanic で同じ score をもう一度出せる」だけではありません。

このリポジトリを手本にして別コンペで `kaggle-template` を使ったときも、以下を再現できることを
目標にしています。

- どの順番で調査・実験・判断するか
- CV を信頼する前に何を監査するか
- YAML 1枚を1実験として何を保存するか
- 特徴量候補をどう作り、CV にどう選ばせるか
- HPO と最終評価をどう分離するか
- 同じ fold の OOF からだけで ensemble weight をどう決めるか
- Public LB を意思決定へ混ぜない方法

`kaggle-template` が小さな実験基盤、このリポジトリがその基盤を使う**具体的な作業手順書**です。
competition 固有の列名・CV 単位・特徴量・探索空間は置き換えますが、判断順序と成果物は変えません。

## ワークフローと残す証拠

| Stage | 実施内容 | 必ず残すもの |
| --- | --- | --- |
| 1 | Simple EDA | 実行済み notebook、データ契約、target 分布、欠損 |
| 2 | First submission | config、OOF、確率 test prediction、submission |
| 3 | Detailed EDA / CV audit | group/time leakage、fold overlap、AV、固定 fold CSV |
| 4 | Diverse baselines | 同一 fold の5モデル OOF、score、importance |
| 5 | Feature Engineering | target-free 候補、block 定義、全 selection trial |
| 6 | HPO | 専用 tuning fold、全 trial、best parameter、通常 YAML |
| 7 | Confirmation CV | HPO best を未使用 seed の固定 fold で再評価した OOF |
| 8 | Ensemble | OOF 相関、探索履歴、weight、threshold、submission |
| 9 | Review | 全実験表と分析 notebook |

## セットアップ

Python 3.10 以上と [uv](https://docs.astral.sh/uv/) を使います。依存関係は `uv.lock` で固定しています。

```bash
git clone --branch spaceship-titanic-workflow https://github.com/fumin0ri/kaggle-example.git
cd kaggle-example
uv sync --extra dev
```

Kaggle の API token を用意し、コンペルールへ同意してから公式 CLI でデータを取得します。

```bash
uv run python scripts/download_data.py
```

```text
input/
├── train.csv
├── test.csv
└── sample_submission.csv
```

入力 CSV と大きな中間出力は Git 管理外です。config、fold、選択履歴、notebook、結果表、最終 submission
は Git に残します。

## 1. Simple EDA と first submission

EDA と分析はすべて `notebooks/` 以下の実行可能 `.ipynb` に置いています。

```bash
uv run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/01_simple_eda.ipynb

uv run python train.py --config configs/exp001_xgb_initial.yaml
```

最初の実験は意図的に単純です。

- shape、列差、ID 重複、型、欠損、target 比率を確認
- 元列から ID と Name だけを除外
- ordinary shuffled StratifiedKFold
- fold 内で imputation / one-hot encoding
- XGBoost、accuracy、threshold 0.5
- `submission.csv` まで作り、パイプライン全体を疎通

この score は checkpoint であり、まだ信頼できる CV とは扱いません。

## 2. Detailed EDA と CV audit

```bash
uv run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/02_detailed_eda_and_cv.ipynb

uv run python make_folds.py
```

Spaceship Titanic の `PassengerId=gggg_pp` では `gggg` が旅行 group です。通常の StratifiedKFold
は同じ group を学習・検証の両方へ入れるため、以後は `StratifiedGroupKFold` を固定します。

Notebook 02 では次を実行しています。

- ordinary CV と group CV の group overlap 比較
- group 内 target consistency
- category ごとの target rate
- train/test の欠損率差
- train=0 / test=1 の adversarial validation OOF AUC
- AV の重要特徴量
- 以後使う CV の決定

別コンペでは group をそのデータの生成単位へ置き換えます。

- 顧客データ: customer ID
- 医療データ: patient ID / hospital
- セッションデータ: user / session
- 時系列: future が train に入らない time split
- 画像: original image / subject / location

「StratifiedKFold を使うこと」ではなく、**未知 test を模倣する分割を決めて固定すること**が再利用する原則です。

## 3. 同じ fold で diverse baseline

```bash
uv run python train.py --config configs/exp101_catboost_baseline.yaml
uv run python train.py --config configs/exp102_xgboost_baseline.yaml
uv run python train.py --config configs/exp103_extratrees_baseline.yaml
uv run python train.py --config configs/exp104_logistic_baseline.yaml
uv run python train.py --config configs/exp105_mlp_baseline.yaml
```

| Model | 役割 |
| --- | --- |
| CatBoost | native category と boosting |
| XGBoost | one-hot 後の nonlinear boosting |
| ExtraTrees | randomized bagging |
| Logistic Regression | linear baseline |
| MLP | smooth nonlinear model |

強いモデルを複製するのではなく、OOF の誤り方が異なる候補を作ります。前処理は sklearn Pipeline
の中で fold ごとに fit されます。

## 4. Feature Engineering at Scale

`kaggle_workflow/features.py` は40個以上の候補を11個の意味のある block で管理します。

- identity
- cabin
- group
- name
- age
- spend total / statistics
- spend log
- spend ratio
- missingness
- categorical cross
- domain consistency rules

個別列を無制限に試すと selection overfitting が強くなるため、block 単位で greedy forward
selection します。

```bash
uv run python select_features.py
```

- 全比較: `artifacts/feature_selection.csv`
- 採用 block: `artifacts/selected_blocks.yaml`
- 選択基準: 固定 group-CV OOF accuracy
- 改善が閾値未満なら停止

採用 block で5モデルを再学習します。

```bash
uv run python train.py --config configs/exp201_catboost_selected.yaml
uv run python train.py --config configs/exp202_xgboost_selected.yaml
uv run python train.py --config configs/exp203_extratrees_selected.yaml
uv run python train.py --config configs/exp204_logistic_selected.yaml
uv run python train.py --config configs/exp205_mlp_selected.yaml
```

target encoding のように目的変数を使う特徴量を追加する場合は、必ず fold 内で fit してください。この例の
group size、surname frequency、cabin occupancy は train+test の target-free 集計です。

## 5. HPO — tuning と confirmation を分ける

HPO は feature selection の後、ensemble の前に行います。実装は `hpo.py` です。

```bash
uv run python hpo.py --model catboost   --n-trials 30
uv run python hpo.py --model xgboost    --n-trials 30
uv run python hpo.py --model extratrees --n-trials 30
uv run python hpo.py --model logistic   --n-trials 20
uv run python hpo.py --model mlp        --n-trials 30
```

重要なのは HPO score を最終 CV score と呼ばないことです。

```text
専用 tuning CV
  StratifiedGroupKFold(3), seed=2025
            │
            ├─ Optuna TPE, sampler seed固定
            ├─ trial 0 = 現在の手動 baseline
            ├─ 全 trial CSV + SQLite study
            └─ best parameter を通常の YAML に確定
                         │
                         ▼
confirmation CV
  StratifiedGroupKFold(5), seed=42, 全モデル共通 fold
            │
            └─ OOF / test probabilities / metrics を通常実験として保存
```

Optuna の study は `artifacts/hpo/*.db` に保存され、中断再開できます。`--n-trials` は追加数ではなく
completed trial の**総予算**として扱うため、同じコマンドを再実行しても勝手に trial が増えません。
search space は `hpo.py::suggest_parameters` に明示し、best parameter は次の通常 config に変換されます。

```text
configs/exp301_catboost_hpo.yaml
configs/exp302_xgboost_hpo.yaml
configs/exp303_extratrees_hpo.yaml
configs/exp304_logistic_hpo.yaml
configs/exp305_mlp_hpo.yaml
```

```bash
uv run python train.py --config configs/exp301_catboost_hpo.yaml
uv run python train.py --config configs/exp302_xgboost_hpo.yaml
uv run python train.py --config configs/exp303_extratrees_hpo.yaml
uv run python train.py --config configs/exp304_logistic_hpo.yaml
uv run python train.py --config configs/exp305_mlp_hpo.yaml
```

HPO best が confirmation CV で悪化した場合、元モデルを置換しません。両方を候補として残し、最終的に
OOF が判断します。これは HPO も CV に過適合するためです。

## 6. OOF Hill Climbing Ensemble

baseline、selected-feature、HPO の15候補を渡します。

```bash
uv run python ensemble.py \
  output/exp101_catboost_baseline output/exp102_xgboost_baseline \
  output/exp103_extratrees_baseline output/exp104_logistic_baseline \
  output/exp105_mlp_baseline \
  output/exp201_catboost_selected output/exp202_xgboost_selected \
  output/exp203_extratrees_selected output/exp204_logistic_selected \
  output/exp205_mlp_selected \
  output/exp301_catboost_hpo output/exp302_xgboost_hpo \
  output/exp303_extratrees_hpo output/exp304_logistic_hpo \
  output/exp305_mlp_hpo
```

Hill Climbing は Public LB を受け取りません。

1. 各候補の PassengerId、target、fold、行順が一致することを検証
2. OOF accuracy 最大の単体から開始
3. 各候補を 1%〜50% 混ぜる weight step と threshold を OOF 上で探索
4. accuracy 同率なら log loss を tie-breaker に使用
5. 改善がなくなったら停止
6. 同じ weight / threshold を test probabilities へ適用

重み 0 は失敗ではありません。OOF が不要と判断したモデルを無理に混ぜないことも ensemble selection
の結果です。

## 7. 最終分析 Notebook

```bash
uv run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/03_hpo_and_ensemble_analysis.ipynb
```

Notebook 03 には次が出力されます。

- 全実験の confirmation CV 比較
- feature selection の全履歴
- モデル別 HPO learning curve
- tuning CV と confirmation CV の区別
- OOF correlation matrix
- Hill Climbing weight と最終 metrics
- 別コンペへ移すときの checklist

## 今回の実行結果

この commit では HPO 機構の実行確認として各モデル5 trial を実走しました。実戦では上記の30 trial
程度から開始し、計算資源と score の収束を見て増やします。

| Experiment | 5-fold group-CV accuracy | OOF AUC |
| --- | ---: | ---: |
| initial XGBoost / ordinary split | 0.793972 | 0.873007 |
| CatBoost basic | **0.821235** | 0.904894 |
| XGBoost basic | 0.815024 | 0.903840 |
| ExtraTrees basic | 0.806626 | 0.887331 |
| Logistic basic | 0.797193 | 0.889680 |
| MLP basic | 0.802715 | 0.895000 |
| CatBoost HPO | 0.816749 | 0.901044 |
| XGBoost HPO | 0.814794 | 0.903586 |
| ExtraTrees HPO | 0.804670 | 0.896810 |
| Logistic HPO | 0.798804 | 0.890802 |
| MLP HPO | 0.806166 | 0.897505 |
| Hill Climbing | **0.821351** | **0.904913** |

実際に HPO で MLP と Logistic は改善しました。一方、CatBoost と ExtraTrees の tuning-CV best は
confirmation CV で悪化しました。元 config を消さず、HPO score を最終 score と誤認しない設計が
機能した例です。

- ordinary split の group overlap: fold 当たり 572〜604
- fixed group split の overlap: 全 fold 0
- adversarial validation OOF AUC: 0.5052
- 最終非ゼロ weight: CatBoost basic 0.9915、CatBoost selected 0.0085
- HPO候補の最終 weight: 0（OOF が採用せず）

## 全工程を一括実行

```bash
# 実戦向けデフォルト: 各モデル total 30 trials
uv run python run_all.py

# 機構確認用の短い run
uv run python run_all.py --hpo-trials 5
```

実行順は notebook 01 → first submission → notebook 02 → fixed folds → 5 baselines → feature
selection → selected-feature 5 models → HPO → HPO 5 modelsの confirmation → ensemble → notebook 03
です。

## 別コンペへ同じフローを移す

まず新しいリポジトリを template から作ります。

```bash
git clone https://github.com/fumin0ri/kaggle-template.git my-new-competition
cd my-new-competition
uv sync
```

その後、このリポジトリを参照しながら次の順で置き換えます。

### そのまま再利用する考え方・構造

- `configs/expXXX.yaml` による1実験1設定
- fold ID の保存と全モデル共有
- fold-local preprocessing
- `oof.csv` / `test_predictions.csv` / `submission.csv` / `metrics.json`
- feature block selection
- HPO study / trial CSV / best config の分離
- tuning CV と confirmation CV の分離
- OOF Hill Climbing と行順 validation
- `results.csv` と notebook による意思決定記録

### コンペごとに必ず決め直すもの

| 場所 | 置き換える内容 |
| --- | --- |
| Notebook 01 | target、ID、metric、データ契約 |
| Notebook 02 | group/time split、leakage 仮説、AV 特徴 |
| `make_folds.py` / `cv.py` | 未知 test を模倣する分割単位 |
| `features.py` | domain knowledge に基づく target-free block |
| baseline YAML | metric と妥当な初期 parameter |
| `hpo.py` | 各モデルの探索範囲と計算 budget |
| `train.py` | multiclass/regression/ranking 等なら予測・metric 部分 |
| `ensemble.py` | competition metric と threshold の要否 |
| Notebook 03 | 採否基準、stability、OOF correlation の解釈 |

### 別コンペ開始時の判断チェックリスト

1. submission schema と metric をコードで検証したか
2. 最初の submission を最小構成で作ったか
3. row-level random CV が本当に未知 test を模倣するか
4. group/time/entity leakage を監査したか
5. fold を固定し、全モデルの OOF 行を揃えたか
6. linear / boosting / bagging / neural の異なる baseline を残したか
7. 特徴量候補を意味のある block として比較したか
8. HPO 用 CV と confirmation CV を分けたか
9. HPO 前のモデルを消さず confirmation score で比較したか
10. ensemble weight と threshold を OOF だけで決めたか
11. Public LB を見て config を後付け変更していないか
12. config、fold、OOF、trial、notebook、submission が再生成可能か

## 検証

```bash
uv run ruff check .
uv run pytest
```

テストでは feature schema、group overlap、threshold / hill climb、HPO parameter materialization、
notebook JSON、submission schema を確認します。

## 構成

```text
.
├── configs/                  # baseline / selected / HPO best YAML
├── notebooks/                # EDA・CV監査・結果分析の実行済み ipynb
├── kaggle_workflow/          # features / CV / model / metrics
├── artifacts/
│   ├── folds_*.csv
│   ├── feature_selection.csv
│   ├── selected_blocks.yaml
│   └── hpo/                  # trial CSV / best JSON（DBはGit管理外）
├── scripts/download_data.py
├── train.py
├── hpo.py
├── select_features.py
├── ensemble.py
├── run_all.py
├── results.csv
├── submissions/
├── tests/
├── pyproject.toml
└── uv.lock
```

最も重要なのは、モデルを複雑にすることではなく、**速く提出し、CV を監査し、変更を OOF で比較し、
判断の証拠を残す順番**を別コンペでも繰り返すことです。
