# Datasets

Only small files are committed. Everything else is downloaded locally and ignored by git
(see `.gitignore`).

| File / folder | Source | In git? |
|---|---|---|
| `youtoxic_english_1000.csv` | [YouTube Toxicity Data](https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data) (label: `IsToxic`) | yes (small) |
| Jigsaw Toxic Comment Classification (Wikipedia talk pages) | [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) | no |
| Jigsaw Unintended Bias / Civil Comments (news comments) | [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) | no |
| GOSU.ai English Dota 2 chat | [Kaggle dataset](https://www.kaggle.com/datasets/danielfesalbon/gosu-ai-english-dota-chat) | no |
| r/jobs comments and the hand-labelled test set | collected by the team | no |

Label mapping to toxic / non-toxic is described in the project proposal (`Project Proposal/`).
Put new downloads in this folder; they stay untracked automatically.

`notebook/3_pooled_data.ipynb` expects the Kaggle downloads in folders named after their Kaggle
dataset / competition; only the files listed are read:

```
dataset/
  youtoxic_english_1000.csv
  jigsaw-toxic-comment-classification-challenge/train.csv.zip      (read as zip)
  jigsaw-unintended-bias-in-toxicity-classification/train.csv
  gosu-ai-english-dota-chat/tagged-data.csv
  pooled/                                                         (created by notebook 3)
```
