# Spaceship Titanic: 再現可能な Kaggle 実践フロー

`kaggle-template` の「YAML 1枚 = 1実験」「CV / OOF / submission / 実験表を必ず残す」方針を使い、Spaceship Titanic を最初の提出から OOF Hill Climbing Ensemble まで一通り実践するリポジトリです。

Public LB はモデル・特徴量・重み・閾値の選択に使いません。コンペ指標は **accuracy** です。

## 最終成果物

- `submissions/submission_initial_xgb.csv`: 最初の XGBoost 提出
- `submissions/submission_hill_climb.csv`: 最終 OOF Hill Climbing 提出
- `results.csv`: 全実験の CV accuracy / AUC
- `reports/01_simple_eda.md`: 初期 EDA
- `reports/02_detailed_eda.md`: 詳細 EDA、CV 監査、adversarial validation
- `output/<experiment>/oof.csv`: 各モデルの OOF 確率（再生成物、Git 管理外）
- `output/hill_climb_ensemble/weights.csv`: OOF だけで決めた重み（再生成物）

## 0. セットアップとデータ

Python 3.10 以上と [uv](https://docs.astral.sh/uv/) を使います。

```bash
git clone --branch spaceship-titanic-workflow https://github.com/fumin0ri/kaggle-example.git
cd kaggle-example
uv sync --extra dev
```

Kaggle の Settings で API token を準備し、コンペルールへ同意してから公式 CLI で取得します。

```bash
uv run python scripts/download_data.py
```

結果は次の配置になります。入力 CSV は Git に含めません。

```text
input/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## 1. 簡単な EDA → XGBoost の最初の提出

```bash
uv run python eda.py --level simple
uv run python train.py --config configs/exp001_xgb_initial.yaml
```

最初は意図的に単純です。

- shape、型、欠損率、目的変数比率を確認
- 元列だけ（ID と Name は除外）
- ordinary shuffled `StratifiedKFold(n_splits=5, seed=42)`
- fold 内で欠損補完 + one-hot encoding
- XGBoost、閾値 0.5
- `output/exp001_xgb_initial/submission.csv` を生成

ここは「パイプラインが最後まで通る」ことを最優先にした checkpoint で、まだ CV を信頼しません。

## 2. 詳細 EDA → CV を信頼できる形へ

```bash
uv run python eda.py --level detailed
uv run python make_folds.py
```

`PassengerId` は `gggg_pp` で、同じ `gggg` の乗客は一緒に旅行しています。通常の StratifiedKFold では同じ group が学習 fold と検証 fold に跨り、group 内で似た目的変数を持つ性質を間接的に利用できます。

そこで以後は次を固定します。

- `StratifiedGroupKFold`: stratify は `Transported`、group は PassengerId の `gggg`
- `artifacts/folds_group_5_seed42.csv` を全モデルが読み込む
- 各 fold の group overlap が 0 であることを監査
- 各モデルで同じ行・同じ fold の OOF を作る

詳細 EDA はさらに以下を保存します。

- カテゴリ別 target rate、group size と target の関係
- train/test の欠損率差
- train/test を判別する adversarial validation の 5-fold OOF AUC
- AV の重要特徴量

AV AUC が 0.5 より大きければ分布差があります。AV は警報器であり、特徴量を採用する根拠は最終的に group CV の再現可能な改善です。

## 3. 多様な 5 baseline

```bash
uv run python train.py --config configs/exp101_catboost_baseline.yaml
uv run python train.py --config configs/exp102_xgboost_baseline.yaml
uv run python train.py --config configs/exp103_extratrees_baseline.yaml
uv run python train.py --config configs/exp104_logistic_baseline.yaml
uv run python train.py --config configs/exp105_mlp_baseline.yaml
```

| Model | 主な帰納バイアス | 前処理 |
| --- | --- | --- |
| CatBoost | ordered boosting、カテゴリに強い | native categorical |
| XGBoost | boosted trees、非線形相互作用 | fold-local one-hot |
| ExtraTrees | randomized bagging | fold-local one-hot |
| Logistic Regression | linear baseline | one-hot + scaling |
| MLP | smooth nonlinear interactions | dense one-hot + scaling |

「強いモデルを5個」ではなく、OOF 誤りの相関が異なる候補を作ることが目的です。全モデルは同じ group fold を使います。

## 4. Feature Engineering at Scale

`kaggle_workflow/features.py` は 40 個以上の target-free 候補を 11 block で管理します。

- identity: group 番号、group 内番号
- cabin: deck / number / side / occupancy / band
- group: group size、solo、leader
- name: name length、surname frequency
- age: band、child/minor/senior、二乗
- spend_total: 合計、用途別合計、分散、利用数、no-spend
- spend_log: 各支出と合計の `log1p`
- spend_ratio: 支出 share、age/group 当たり支出
- missing: 欠損個数・欠損フラグ
- cross: route、planet × deck、route × side
- rules: cryosleep × spend、VIP × spend、child × cryosleep

個々の列を無制限に試すと selection overfitting するため、意味のある block 単位で greedy forward selection します。評価器は固定 StratifiedGroupKFold の ExtraTrees、目的は OOF accuracy です。

```bash
uv run python select_features.py
```

全 trial は `artifacts/feature_selection.csv`、採用 block は `artifacts/selected_blocks.yaml` に残ります。改善が 0.0001 未満なら停止します。採用結果を使って5モデルを再学習します。

```bash
uv run python train.py --config configs/exp201_catboost_selected.yaml
uv run python train.py --config configs/exp202_xgboost_selected.yaml
uv run python train.py --config configs/exp203_extratrees_selected.yaml
uv run python train.py --config configs/exp204_logistic_selected.yaml
uv run python train.py --config configs/exp205_mlp_selected.yaml
```

頻度・group size・cabin occupancy は train+test を結合して計算しますが、目的変数は一切使いません。これは transductive な unsupervised feature です。目的変数を使う集約を追加する場合は必ず fold 内で fit してください。

## 5. OOF Hill Climbing Ensemble

```bash
uv run python ensemble.py \
  output/exp101_catboost_baseline \
  output/exp102_xgboost_baseline \
  output/exp103_extratrees_baseline \
  output/exp104_logistic_baseline \
  output/exp105_mlp_baseline \
  output/exp201_catboost_selected \
  output/exp202_xgboost_selected \
  output/exp203_extratrees_selected \
  output/exp204_logistic_selected \
  output/exp205_mlp_selected
```

手順は次の通りです。

1. OOF accuracy が最大の単体モデルから開始（同率なら log loss が小さい方）
2. 各候補を 5%〜50% 混ぜる weight step を全探索し、accuracy、次に log loss の順で選択
3. 改善がある間だけ反復（同じモデルの複数回選択を許す）
4. 採用した step を再帰的に convex weight へ変換
5. 最終 OOF だけで competition metric の accuracy が最大になる threshold を探索
6. 同じ重み・閾値を test prediction に適用

各候補 blend の threshold も OOF だけで探索します。accuracy が同率のときは滑らかな log loss を
tie-breaker にして plateau 上でも校正の良い組合せを選びます。baseline と selected-feature の
両方を候補にし、悪い候補を採用するかどうかも OOF に決めさせます。
`ensemble.py` は LB score を入力として受け取りません。重み、履歴、OOF、最終提出は
`output/hill_climb_ensemble/` に保存されます。

## 今回の再現実行結果

Python 3.12、seed 42 で実際に全段階を走らせた結果です。ライブラリ版で末尾は多少変わり得ます。

| Experiment | OOF accuracy | OOF AUC |
| --- | ---: | ---: |
| initial XGBoost / ordinary stratified | 0.793972 | 0.873007 |
| CatBoost basic / group CV | 0.821235 | 0.904894 |
| XGBoost basic / group CV | 0.815024 | 0.903840 |
| ExtraTrees basic / group CV | 0.806626 | 0.887331 |
| Logistic basic / group CV | 0.797193 | 0.889680 |
| MLP basic / group CV | 0.802715 | 0.895000 |
| ExtraTrees selected features / group CV | 0.809272 | 0.891099 |
| Hill climb | **0.821351** | **0.904913** |

詳細 EDA の AV AUC は 0.5052、通常 split の group overlap は fold 当たり 572〜604、group
split はすべて 0 でした。Feature selection は `cabin, group, age, spend_total, missing` に
`identity, name` を追加しました。

Hill climb の非ゼロ重みは CatBoost basic 0.9915、CatBoost selected 0.0085、OOF threshold
0.5 でした。他モデルを無理に混ぜず OOF が不要と判断した重みを 0 にすることも、この手順の重要な
結果です。

## 全部を順番に実行

```bash
uv run python run_all.py
```

実行順は `simple EDA → initial submission → detailed EDA/CV audit → 5 baseline → feature selection → selected-feature 5 models → OOF ensemble` です。

## 再現性チェック

```bash
uv run ruff check .
uv run pytest
```

乱数 seed、fold、config、OOF、確率 test prediction、metrics、feature importance を保存します。再実行時は同じ fold CSV を検証して使い、PassengerId の行順が違えば停止します。

## よくある落とし穴

- Public LB を見ながら特徴量・重みを変える: public test への過学習になる
- 5モデルで別々の fold を使う: OOF blend を公平に比較できない
- 全データで scaler / imputer を fit: CV leakage になる。本実装は Pipeline 内で fold ごとに fit
- target encoding を全 train で作る: 強い leakage。本実装の集約は target-free
- accuracy なのに確率の AUC だけを見る: threshold を含む意思決定を評価できない
- submission を 0/1 で出す: 本コンペは `True` / `False` の bool 列で出力する

## 構成

```text
.
├── configs/                  # YAML 1枚 = 1実験
├── kaggle_workflow/          # features / CV / model / metric
├── scripts/download_data.py
├── tests/
├── eda.py
├── make_folds.py
├── train.py
├── select_features.py
├── ensemble.py
├── run_all.py
├── reports/
├── submissions/
└── results.csv
```

まず守るべきなのは、速く提出してから CV を監査し、全変更を OOF で比較できる状態を保つことです。
