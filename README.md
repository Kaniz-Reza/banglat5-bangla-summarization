# Bangla News Summarization with BanglaT5

Fine-tuning [`csebuetnlp/banglat5`](https://huggingface.co/csebuetnlp/banglat5) on the Bengali part of [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) to write a one-sentence summary of a Bangla news article.

The whole project is one notebook, `notebooks/bangla-news-summarization.ipynb`, run on Kaggle (2x Tesla T4) with all outputs kept: data checks, tokenizer checks, baseline, training, decoding experiments, final test evaluation and training curves.

## Results

Test set (1,012 articles), beam search with 8 beams. The test set was used once, after all settings had been chosen on the validation set.

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-22 baseline (first 22 words of the article) | 19.90 | 6.43 | 15.74 |
| **Fine-tuned BanglaT5 (this project)** | **29.41** | **12.28** | **24.99** |

The model is about 9 ROUGE-L points above the Lead-22 baseline. 22 words is the median length of a reference summary, so the baseline is a fair "no model" reference.

For orientation only: the XL-Sum README lists 29.57 / 12.11 / 25.13 for multilingual mT5 on the same Bengali test split, and the BanglaNLG paper reports ROUGE-2 13.7 for BanglaT5. Those numbers come from other training and decoding setups and were not reproduced here, so this is a sanity check that the scores are in the same range, not a head-to-head comparison.

Validation set (1,012 articles):

| Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-22 baseline | 20.47 | 6.61 | 16.12 |
| Greedy | 28.24 | 10.97 | 24.14 |
| Beam search, 8 beams (final) | 29.94 | 12.96 | 25.66 |

ROUGE is computed with the `rouge_score` package patched for Bengali (from the XL-Sum repository), using its Bengali stemmer.

## Method

- **Data:** XL-Sum Bengali, 8,102 train / 1,012 validation / 1,012 test articles. Input is cut at 1,024 tokens, summaries at 96 tokens.
- **Preprocessing:** the `normalizer` package from csebuetnlp is applied to every article and summary before tokenizing.
- **Model:** `csebuetnlp/banglat5` (247.6M parameters), trained in fp32. The T4 GPUs have no bf16, and T5 models often produce NaN in fp16.
- **Training:** Hugging Face `Seq2SeqTrainer`, effective batch size 16 (2 per device x 2 GPUs x 4 accumulation steps), learning rate 1e-4, 10% warm-up, 5 epochs (2,535 steps, about 180 minutes). The best epoch is chosen by validation ROUGE-L, with early stopping (patience 2). The last epoch was the best one.
- **Decoding (final):** `num_beams=8`, `length_penalty=1.0`, `no_repeat_ngram_size=3`, `max_new_tokens=96`. Chosen on the validation set.

| Epoch | Validation loss | ROUGE-L (greedy) |
|---|---|---|
| 1 | 2.732 | 18.47 |
| 2 | 2.627 | 22.55 |
| 3 | 2.587 | 23.37 |
| 4 | 2.567 | 23.57 |
| 5 | 2.563 | 24.14 |

Validation was still improving slowly at epoch 5, so a little more training would probably help. This was not tested.

## A bug that nearly ruined the first run

My first training run reached only 19.58 ROUGE-L on validation, and every summary started with the same word. The cause was not the training settings.

With `transformers` 5.0.0, BanglaT5 loads with `config.scale_decoder_outputs = True`. The T5 forward pass then divides the decoder output by sqrt(768) = 27.7, which squashes all logits. I found it by testing instead of guessing:

- Flipping the flag changed the spread of the logits by exactly 27.7x.
- On the untouched base model, a fill-in-the-blank test gave a loss of 9.78 with the flag on and 3.29 with it off, with the same share of right guesses.

The fix is one line after every model load: `model.config.scale_decoder_outputs = False`. With the same recipe (5 epochs instead of 6), the second run reached 24.14 ROUGE-L on validation (greedy), against 19.58 before.

Two more things in `transformers` 5.0.0 needed care:

- The logged loss and the gradient of the `Trainer` were inflated by (number of GPUs x accumulation steps) for this model. Fix: `trainer.model_accepts_loss_kwargs = False`.
- The embeddings load untied (296.9M parameters instead of 247.6M), so they are tied again after loading.

## Decoding experiments

All on the first 200 validation articles, `no_repeat_ngram_size=3`. Picking the best of several settings on a subset makes these numbers a little optimistic; the full 1,012-article numbers above are the honest ones.

| Setting | ROUGE-1 | ROUGE-2 | ROUGE-L | Average words |
|---|---|---|---|---|
| Greedy | 28.03 | 11.45 | 23.79 | 14.2 |
| Beam 4, length penalty 1.0 | 29.29 | 13.17 | 25.26 | 15.4 |
| Beam 4, length penalty 2.0 | 29.45 | 12.86 | 25.09 | 18.1 |
| Beam 8, length penalty 1.0 (chosen) | 30.46 | 14.14 | 26.01 | 15.8 |
| Beam 8, length penalty 2.0 | 29.72 | 13.15 | 24.98 | 20.5 |
| Human reference | | | | 21.1 |

A higher length penalty gives human-length summaries but lowers ROUGE-L.

## Limitations

- **Wrong first word.** On the test set, 68.5% of the summaries start with "বাংলাদেশ" (human summaries: 23.4%). The model often guesses a country instead of copying it from the article. For example, an article about missing Saudi princes in Europe became "বাংলাদেশে সৌদি রাজপরিবারের তিনজন সদস্যকে অপহরণ করে ..." (greedy decoding, validation set).
- **I tried to fix it and it did not work well enough.** A rule that blocks "বাংলাদেশ" as the first word when the article never mentions it removed the Bangladesh opener in those articles, but it only cut the first words that do not appear in the article by 30.5% (the limit I had set beforehand was one third). The model just swapped in another guessed country (India, the US, the UK), so the rule is not part of the final system.
- **Wrong facts.** Summaries are fluent, but facts can be wrong, and ROUGE does not catch this. On the test set, one summary names the US human rights committee where the reference says the UN, and another drops the number of patients. Do not use the output without checking it.
- **Too short.** Average length is 15.7 words, against 21.6 for the human summaries.

## Reproduce

Tested on Kaggle's older notebook environment (Python 3.12.13) with 2x Tesla T4 and Internet turned on. The newest Kaggle environment uses Python 3.13, where `pyonmttok` (needed by the Bengali stemmer) has no wheel, so the install cell fails there.

1. Create a Kaggle notebook, choose GPU T4 x2, turn Internet on.
2. Upload `notebooks/bangla-news-summarization.ipynb` and run the cells in order. The first cells install the same packages that are listed in `requirements.txt`.
3. Keep these versions: `transformers` 5.0.0 and `datasets` below 4.0 (the dataset is loaded with `trust_remote_code=True`).

Full training takes about 3 hours on 2x T4.

## Repository layout

```
notebooks/bangla-news-summarization.ipynb   all code and outputs
requirements.txt                            packages used
README.md
```

## Model weights and demo

The fine-tuned weights are stored in a private Hugging Face repository and are not public yet. A Hugging Face Spaces demo is planned. Anyone who builds it must load the model with `transformers` 5.0.0 and set `model.config.scale_decoder_outputs = False` after loading, or the output will be wrong.

## Licences and references

The dataset and the base model have their own licences (see their Hugging Face pages). Check them before any commercial use.

- Hasan et al., *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages* (Findings of ACL 2021).
- Bhattacharjee et al., *BanglaNLG and BanglaT5: Benchmarks and Resources for Evaluating Low-Resource Natural Language Generation in Bangla* (Findings of EACL 2023).
