n: {'calibration': 11187, 'validation': 11233, 'generators': 220}


## All slots

| method | CV NLL (cal) | validation NLL | validation ECE | validation Brier | validation acc | generators NLL | generators ECE | generators Brier | generators acc |
|---|---|---|---|---|---|---|---|---|---|
| per type (old artifact) | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.8076 | 0.3838 | 0.0545 | 0.1987 | 0.8081 |
| feature T (thermometer-lite) | 0.4474 | 0.4452 | 0.0121 | 0.2140 | 0.8077 | 0.3821 | 0.0415 | 0.1959 | 0.8081 |
| checkpoint 2 alone, per type T | 0.4278 | 0.4270 | 0.0076 | 0.1997 | 0.8259 | 0.5159 | 0.1176 | 0.2371 | 0.7981 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.4175 | 0.4157 | 0.0129 | 0.1979 | 0.8238 | 0.3824 | 0.0617 | 0.2002 | 0.8179 |

## validation: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| per type (old artifact) | 0.4877 | 0.0179 | 0.3042 | 0.0245 | 0.2711 | 0.4248 | 0.9624 |
| feature T (thermometer-lite) | 0.4871 | 0.0126 | 0.3004 | 0.0158 | 0.2686 | 0.4241 | 0.9625 |
| checkpoint 2 alone, per type T | 0.4605 | 0.0094 | 0.3111 | 0.0160 | 0.2638 | 0.4002 | 0.9244 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.4532 | 0.0142 | 0.2860 | 0.0114 | 0.2569 | 0.3806 | 0.9236 |

## validation: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| per type (old artifact) | 0.2373 | 0.7287 | 0.9554 | 0.0001 | -0.0001 |
| feature T (thermometer-lite) | 0.2724 | 0.7011 | 0.9635 | 0.0000 | 0.0000 |
| checkpoint 2 alone, per type T | 0.2366 | 0.7349 | 0.9627 | 0.1016 | 0.0182 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.2488 | 0.7264 | 0.9670 | 0.0588 | 0.0162 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: none

## generators: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| per type (old artifact) | 0.2978 | 0.0468 | 0.4763 | 0.0876 | 0.4367 | 0.4030 | 0.0234 |
| feature T (thermometer-lite) | 0.2953 | 0.0373 | 0.4754 | 0.0686 | 0.4330 | 0.4020 | 0.0274 |
| checkpoint 2 alone, per type T | 0.3247 | 0.0739 | 0.7214 | 0.1650 | 0.4681 | 0.6547 | 0.0604 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.2852 | 0.0598 | 0.4869 | 0.0720 | 0.4412 | 0.3921 | 0.0394 |

## generators: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| per type (old artifact) | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.4400 | 0.5300 | 0.9464 | 0.0000 | 0.0000 |
| checkpoint 2 alone, per type T | 0.4000 | 0.5500 | 0.9167 | 0.0909 | -0.0101 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.4300 | 0.5400 | 0.9474 | 0.0682 | 0.0098 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: none

## Oracle (fitted on the evaluated set)

| method | NLL | ECE | en NLL | fr NLL |
|---|---|---|---|---|
| validation type | 0.4465 | 0.0182 | 0.4878 | 0.3037 |
| validation type_lang | 0.4460 | 0.0174 | 0.4876 | 0.3019 |
| generators type | 0.3743 | 0.0410 | 0.2858 | 0.4693 |
| generators type_lang | 0.3737 | 0.0463 | 0.2850 | 0.4690 |

## Guessed language (rows: tagged)

```
{
 "calibration": {
  "en": {
   "?": 983,
   "en": 7709,
   "fr": 2
  },
  "fr": {
   "?": 98,
   "fr": 2395
  }
 },
 "validation": {
  "en": {
   "?": 1030,
   "en": 7682,
   "fr": 1
  },
  "fr": {
   "?": 102,
   "fr": 2418
  }
 },
 "generators": {
  "en": {
   "?": 30,
   "en": 84
  },
  "fr": {
   "?": 9,
   "fr": 97
  }
 }
}
```

## Fitted parameters

```
{
 "per type (old artifact)": {
  "temperatures": {
   "choice": 1.1332639044221633,
   "noul": 1.3120353701728706,
   "score": 1.062142349167514
  },
  "tau": null,
  "cv_nll": null
 },
 "feature T (thermometer-lite)": {
  "type_log_t": {
   "choice": 0.18546833034450363,
   "noul": 0.13317284463023546,
   "score": 0.15036555483852973
  },
  "w": [
   -0.06975907755648683,
   0.0006784728899259903,
   -0.035835788984718175,
   -0.03898917837704973,
   0.2605691420018935,
   0.22438436723582278
  ],
  "features": [
   "ln_options",
   "lang_fr",
   "lang_unknown",
   "ln_tokens",
   "entropy",
   "max_prob"
  ],
  "lambda": 1.0,
  "cv_nll": {
   "0.1": 0.4470635285685435,
   "1.0": 0.44706251423154986,
   "10.0": 0.44706505578780126,
   "100.0": 0.4473050304113106,
   "1000.0": 0.4481340871665978,
   "10000.0": 0.44906191308703486,
   "inf": 0.4493400982043274
  }
 },
 "checkpoint 2 alone, per type T": {},
 "checkpoints 1 + 2, mean of calibrated probabilities": {}
}
```
