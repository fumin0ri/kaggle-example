# 02 Detailed EDA and CV audit

## Main findings

- Passenger groups: 6,217; groups whose observed members all share one target: 5,420.
- Ordinary StratifiedKFold group overlap per fold: [593, 604, 590, 582, 572].
- StratifiedGroupKFold group overlap per fold: [0, 0, 0, 0, 0].
- Adversarial-validation OOF ROC AUC: **0.5052**.
  (0.5 means no detectable train/test drift.)

The ordinary split is useful only as the requested first checkpoint. Since members of the same
PassengerId group often share outcomes, its train/validation overlap makes it optimistic. Every
later experiment therefore loads the exact same persisted StratifiedGroupKFold assignment.

Adversarial validation is diagnostic, not a Public-LB tuning signal. A high AUC means that some
features expose the chronological/train-test construction split; inspect
`adversarial_importance.csv` and require improvements to hold under group CV.

## Fold audit

| fold | stratified_target_rate | group_target_rate | stratified_group_overlap | group_group_overlap |
| --- | --- | --- | --- | --- |
| 0 | 0.503737780333525 | 0.503737780333525 | 593 | 0 |
| 1 | 0.503737780333525 | 0.503737780333525 | 604 | 0 |
| 2 | 0.503737780333525 | 0.503737780333525 | 590 | 0 |
| 3 | 0.503452243958573 | 0.503452243958573 | 582 | 0 |
| 4 | 0.503452243958573 | 0.503452243958573 | 572 | 0 |

## Target rates for important categorical variables

| feature | value | target_rate | count |
| --- | --- | --- | --- |
| HomePlanet | Earth | 0.42394611038678837 | 4602 |
| HomePlanet | Europa | 0.65884561238855 | 2131 |
| HomePlanet | Mars | 0.5230244457077885 | 1759 |
| HomePlanet | __MISSING__ | 0.5124378109452736 | 201 |
| CryoSleep | False | 0.3289207574921861 | 5439 |
| CryoSleep | True | 0.8175831412578202 | 3037 |
| CryoSleep | __MISSING__ | 0.48847926267281105 | 217 |
| Destination | TRAPPIST-1e | 0.47117497886728654 | 5915 |
| Destination | 55 Cancri e | 0.61 | 1800 |
| Destination | PSO J318.5-22 | 0.5037688442211056 | 796 |
| Destination | __MISSING__ | 0.5054945054945055 | 182 |
| VIP | False | 0.5063321674104451 | 8291 |
| VIP | __MISSING__ | 0.5123152709359606 | 203 |
| VIP | True | 0.38190954773869346 | 199 |

## Group-size behavior

| group_size | mean | std | count |
| --- | --- | --- | --- |
| 1 | 0.45244536940686786 | 0.4977852210885682 | 4805 |
| 2 | 0.5380499405469679 | 0.3704604629995351 | 841 |
| 3 | 0.5931372549019608 | 0.30426654849207774 | 340 |
| 4 | 0.6407766990291263 | 0.2681406176324111 | 103 |
| 5 | 0.5924528301886793 | 0.2540813441852625 | 53 |
| 6 | 0.6149425287356322 | 0.27132628499335953 | 29 |
| 7 | 0.5411255411255411 | 0.2223413667409976 | 33 |
| 8 | 0.3942307692307692 | 0.16012815380508713 | 13 |
