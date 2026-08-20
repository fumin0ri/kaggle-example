# 01 Simple EDA

## Shape

- train: 8,693 rows x 14 columns
- test: 4,277 rows x 13 columns
- duplicated PassengerId in train/test: 0 / 0

## Target distribution

| Transported | count | rate |
| --- | --- | --- |
| True | 4378 | 0.5036236051995858 |
| False | 4315 | 0.4963763948004141 |

## Missing values and dtypes

| column | dtype | train_missing | train_missing_rate | test_missing |
| --- | --- | --- | --- | --- |
| PassengerId | str | 0 | 0.0 | 0.0 |
| HomePlanet | str | 201 | 0.023122052225928908 | 87.0 |
| CryoSleep | object | 217 | 0.02496261359714713 | 93.0 |
| Cabin | str | 199 | 0.02289198205452663 | 100.0 |
| Destination | str | 182 | 0.02093638559760727 | 92.0 |
| Age | float64 | 179 | 0.020591280340503854 | 91.0 |
| VIP | object | 203 | 0.023352122397331185 | 93.0 |
| RoomService | float64 | 181 | 0.02082135051190613 | 82.0 |
| FoodCourt | float64 | 183 | 0.021051420683308408 | 106.0 |
| ShoppingMall | float64 | 208 | 0.02392729782583688 | 98.0 |
| Spa | float64 | 183 | 0.021051420683308408 | 101.0 |
| VRDeck | float64 | 188 | 0.021626596111814105 | 80.0 |
| Name | str | 200 | 0.023007017140227768 | 94.0 |
| Transported | bool | 0 | 0.0 |  |

## First baseline decision

The target is nearly balanced, so the first reproducible baseline uses accuracy, a fixed 0.5
threshold, ordinary shuffled StratifiedKFold, one-hot preprocessing, and XGBoost. This is a
deliberately simple checkpoint; group leakage is audited before trusting later CV scores.
