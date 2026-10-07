n: {'calibration': 11187, 'validation': 11233, 'generators': 220}


## All slots

| method | CV NLL (cal) | validation NLL | validation ECE | validation Brier | validation acc | generators NLL | generators ECE | generators Brier | generators acc |
|---|---|---|---|---|---|---|---|---|---|
| per type (old artifact) | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.8076 | 0.3838 | 0.0545 | 0.1987 | 0.8081 |
| feature T (thermometer-lite) | 0.4474 | 0.4452 | 0.0121 | 0.2140 | 0.8077 | 0.3821 | 0.0415 | 0.1959 | 0.8081 |
| checkpoint 2 alone, per type T | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.8170 | 0.3892 | 0.0585 | 0.1886 | 0.8024 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.4353 | 0.4328 | 0.0129 | 0.2078 | 0.8135 | 0.3646 | 0.0449 | 0.1849 | 0.8182 |

## validation: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| per type (old artifact) | 0.4877 | 0.0179 | 0.3042 | 0.0245 | 0.2711 | 0.4248 | 0.9624 |
| feature T (thermometer-lite) | 0.4871 | 0.0126 | 0.3004 | 0.0158 | 0.2686 | 0.4241 | 0.9625 |
| checkpoint 2 alone, per type T | 0.4689 | 0.0147 | 0.2943 | 0.0174 | 0.2708 | 0.3976 | 0.9304 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.4725 | 0.0136 | 0.2954 | 0.0210 | 0.2677 | 0.4053 | 0.9368 |

## validation: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| per type (old artifact) | 0.2373 | 0.7287 | 0.9554 | 0.0001 | -0.0001 |
| feature T (thermometer-lite) | 0.2724 | 0.7011 | 0.9635 | 0.0000 | 0.0000 |
| checkpoint 2 alone, per type T | 0.2272 | 0.7373 | 0.9540 | 0.0637 | 0.0093 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.2313 | 0.7347 | 0.9557 | 0.0339 | 0.0058 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: none

## generators: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| per type (old artifact) | 0.2978 | 0.0468 | 0.4763 | 0.0876 | 0.4367 | 0.4030 | 0.0234 |
| feature T (thermometer-lite) | 0.2953 | 0.0373 | 0.4754 | 0.0686 | 0.4330 | 0.4020 | 0.0274 |
| checkpoint 2 alone, per type T | 0.3659 | 0.0551 | 0.4143 | 0.0696 | 0.4381 | 0.4057 | 0.0625 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.3031 | 0.0400 | 0.4307 | 0.0681 | 0.4272 | 0.3667 | 0.0408 |

## generators: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| per type (old artifact) | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.4400 | 0.5300 | 0.9464 | 0.0000 | 0.0000 |
| checkpoint 2 alone, per type T | 0.4200 | 0.5400 | 0.9310 | 0.1273 | -0.0057 |
| checkpoints 1 + 2, mean of calibrated probabilities | 0.4200 | 0.5400 | 0.9310 | 0.0773 | 0.0101 |

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
