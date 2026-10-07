n: {'calibration': 11187, 'validation': 11233, 'generators': 220}


## All slots

| method | CV NLL (cal) | validation NLL | validation ECE | validation Brier | validation acc | generators NLL | generators ECE | generators Brier | generators acc |
|---|---|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4467 | 0.4458 | 0.0359 | 0.2030 | 0.8259 | 0.6507 | 0.1321 | 0.2581 | 0.7981 |
| global T | 0.4308 | 0.4292 | 0.0105 | 0.1999 | 0.8259 | 0.5281 | 0.1131 | 0.2415 | 0.7981 |
| per type (old artifact) | 0.4278 | 0.4270 | 0.0076 | 0.1997 | 0.8259 | 0.5159 | 0.1176 | 0.2371 | 0.7981 |
| per lang | 0.4303 | 0.4280 | 0.0117 | 0.1996 | 0.8259 | 0.5058 | 0.1157 | 0.2376 | 0.7981 |
| per type × lang, unshrunk | 0.4276 | 0.4261 | 0.0076 | 0.1994 | 0.8259 | 0.4961 | 0.1168 | 0.2346 | 0.7981 |
| per type × lang, ridge (CV tau) [impl] | 0.4276 | 0.4261 | 0.0076 | 0.1994 | 0.8259 | 0.4980 | 0.1170 | 0.2349 | 0.7981 |
| per type × lang, ridge, served (guessed lang) | 0.4277 | 0.4263 | 0.0081 | 0.1995 | 0.8259 | 0.4979 | 0.1170 | 0.2349 | 0.7981 |
| per type × lang, empirical Bayes | 0.4277 | 0.4264 | 0.0076 | 0.1995 | 0.8259 | 0.5058 | 0.1219 | 0.2359 | 0.7981 |
| T(n) = a + b ln n | 0.4302 | 0.4290 | 0.0115 | 0.1995 | 0.8259 | 0.5370 | 0.1140 | 0.2405 | 0.7981 |
| T(n) = a_type + b ln n | 0.4282 | 0.4269 | 0.0081 | 0.1997 | 0.8259 | 0.5153 | 0.1176 | 0.2370 | 0.7981 |
| per source (upper bound) | 0.4111 | 0.4098 | 0.0083 | 0.1938 | 0.8259 | 0.3982 | 0.0669 | 0.2071 | 0.7981 |
| feature T (thermometer-lite) | 0.4267 | 0.4254 | 0.0103 | 0.1995 | 0.8259 | 0.4925 | 0.1128 | 0.2346 | 0.7981 |
| vector scaling: T + position bias, choice | 0.4279 | 0.4273 | 0.0073 | 0.1998 | 0.8254 | 0.5140 | 0.1174 | 0.2369 | 0.7981 |
| vector scaling: T + position bias, every type | 0.4275 | 0.4273 | 0.0093 | 0.1999 | 0.8257 | 0.5129 | 0.1170 | 0.2362 | 0.7981 |
| batch calibration (eval batch), choice+noul | 0.4289 | 0.4272 | 0.0068 | 0.1998 | 0.8259 | 0.5136 | 0.1126 | 0.2372 | 0.7981 |
| batch calibration (calibration split), choice+noul | 0.4278 | 0.4271 | 0.0083 | 0.1997 | 0.8259 | 0.5152 | 0.1176 | 0.2370 | 0.7981 |
| per type T + Platt with yes-bias (noul) | 0.4271 | 0.4262 | 0.0084 | 0.1992 | 0.8261 | 0.5160 | 0.0990 | 0.2361 | 0.8094 |
| per type T + beta (noul) | 0.4274 | 0.4263 | 0.0083 | 0.1992 | 0.8262 | 0.5158 | 0.0991 | 0.2362 | 0.8094 |
| orderings: first only, per type T | 0.4298 | 0.4275 | 0.0089 | 0.2000 | 0.8262 | 0.5103 | 0.1027 | 0.2356 | 0.8026 |
| orderings: mean of probabilities, per type T | 0.4278 | 0.4270 | 0.0076 | 0.1997 | 0.8259 | 0.5159 | 0.1176 | 0.2371 | 0.7981 |
| orderings: log-space mean, per type T | 0.4274 | 0.4270 | 0.0085 | 0.1998 | 0.8260 | 0.5150 | 0.1180 | 0.2373 | 0.7981 |
| [diagnostic] per type T + histogram binning | 0.4302 | 0.4299 | 0.0115 | 0.2015 | 0.8251 | 0.5215 | 0.1111 | 0.2397 | 0.7981 |
| [diagnostic] per type T + isotonic (top label) | 0.4324 | 0.4296 | 0.0122 | 0.2004 | 0.8232 | 0.5347 | 0.1099 | 0.2384 | 0.8217 |

## validation: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4746 | 0.0326 | 0.3459 | 0.0497 | 0.2880 | 0.4200 | 0.9264 |
| global T | 0.4631 | 0.0116 | 0.3122 | 0.0250 | 0.2662 | 0.4002 | 0.9324 |
| per type (old artifact) | 0.4605 | 0.0094 | 0.3111 | 0.0160 | 0.2638 | 0.4002 | 0.9244 |
| per lang | 0.4628 | 0.0130 | 0.3079 | 0.0114 | 0.2664 | 0.3980 | 0.9301 |
| per type × lang, unshrunk | 0.4603 | 0.0096 | 0.3076 | 0.0122 | 0.2639 | 0.3980 | 0.9244 |
| per type × lang, ridge (CV tau) [impl] | 0.4603 | 0.0093 | 0.3078 | 0.0114 | 0.2639 | 0.3981 | 0.9244 |
| per type × lang, ridge, served (guessed lang) | 0.4605 | 0.0092 | 0.3081 | 0.0123 | 0.2638 | 0.3986 | 0.9244 |
| per type × lang, empirical Bayes | 0.4603 | 0.0092 | 0.3090 | 0.0129 | 0.2638 | 0.3987 | 0.9244 |
| T(n) = a + b ln n | 0.4626 | 0.0133 | 0.3127 | 0.0187 | 0.2641 | 0.4024 | 0.9303 |
| T(n) = a_type + b ln n | 0.4604 | 0.0100 | 0.3114 | 0.0164 | 0.2639 | 0.4001 | 0.9243 |
| per source (upper bound) | 0.4417 | 0.0108 | 0.2995 | 0.0176 | 0.2503 | 0.3713 | 0.9292 |
| feature T (thermometer-lite) | 0.4590 | 0.0119 | 0.3091 | 0.0183 | 0.2620 | 0.3972 | 0.9270 |
| vector scaling: T + position bias, choice | 0.4607 | 0.0088 | 0.3115 | 0.0171 | 0.2638 | 0.4008 | 0.9244 |
| vector scaling: T + position bias, every type | 0.4608 | 0.0104 | 0.3112 | 0.0156 | 0.2635 | 0.4008 | 0.9253 |
| batch calibration (eval batch), choice+noul | 0.4608 | 0.0091 | 0.3113 | 0.0156 | 0.2638 | 0.4008 | 0.9244 |
| batch calibration (calibration split), choice+noul | 0.4606 | 0.0099 | 0.3113 | 0.0154 | 0.2638 | 0.4004 | 0.9244 |
| per type T + Platt with yes-bias (noul) | 0.4600 | 0.0101 | 0.3094 | 0.0170 | 0.2620 | 0.4002 | 0.9244 |
| per type T + beta (noul) | 0.4600 | 0.0097 | 0.3096 | 0.0159 | 0.2622 | 0.4002 | 0.9244 |
| orderings: first only, per type T | 0.4609 | 0.0118 | 0.3117 | 0.0178 | 0.2640 | 0.4011 | 0.9244 |
| orderings: mean of probabilities, per type T | 0.4605 | 0.0094 | 0.3111 | 0.0160 | 0.2638 | 0.4002 | 0.9244 |
| orderings: log-space mean, per type T | 0.4606 | 0.0089 | 0.3112 | 0.0155 | 0.2640 | 0.4001 | 0.9244 |
| [diagnostic] per type T + histogram binning | 0.4628 | 0.0135 | 0.3163 | 0.0195 | 0.2621 | 0.4051 | 0.9339 |
| [diagnostic] per type T + isotonic (top label) | 0.4632 | 0.0118 | 0.3136 | 0.0180 | 0.2667 | 0.4023 | 0.9278 |

## validation: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| none (T = 1) | 0.1418 | 0.8094 | 0.9432 | 0.0000 | 0.0000 |
| global T | 0.1968 | 0.7681 | 0.9563 | 0.0000 | 0.0000 |
| per type (old artifact) | 0.2366 | 0.7349 | 0.9627 | 0.0000 | 0.0000 |
| per lang | 0.1962 | 0.7689 | 0.9566 | 0.0000 | 0.0000 |
| per type × lang, unshrunk | 0.2377 | 0.7336 | 0.9624 | 0.0000 | 0.0000 |
| per type × lang, ridge (CV tau) [impl] | 0.2377 | 0.7336 | 0.9624 | 0.0000 | 0.0000 |
| per type × lang, ridge, served (guessed lang) | 0.2371 | 0.7343 | 0.9624 | 0.0000 | 0.0000 |
| per type × lang, empirical Bayes | 0.2379 | 0.7332 | 0.9621 | 0.0000 | 0.0000 |
| T(n) = a + b ln n | 0.2122 | 0.7557 | 0.9593 | 0.0000 | 0.0000 |
| T(n) = a_type + b ln n | 0.2373 | 0.7343 | 0.9627 | 0.0000 | 0.0000 |
| per source (upper bound) | 0.2279 | 0.7480 | 0.9687 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.2234 | 0.7467 | 0.9614 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, choice | 0.2366 | 0.7349 | 0.9627 | 0.0044 | -0.0005 |
| vector scaling: T + position bias, every type | 0.2366 | 0.7349 | 0.9627 | 0.0142 | -0.0002 |
| batch calibration (eval batch), choice+noul | 0.2366 | 0.7349 | 0.9627 | 0.0010 | 0.0000 |
| batch calibration (calibration split), choice+noul | 0.2366 | 0.7349 | 0.9627 | 0.0007 | 0.0000 |
| per type T + Platt with yes-bias (noul) | 0.2249 | 0.7439 | 0.9597 | 0.0054 | 0.0003 |
| per type T + beta (noul) | 0.2251 | 0.7435 | 0.9594 | 0.0059 | 0.0003 |
| orderings: first only, per type T | 0.2358 | 0.7356 | 0.9625 | 0.0085 | 0.0004 |
| orderings: mean of probabilities, per type T | 0.2366 | 0.7349 | 0.9627 | 0.0000 | 0.0000 |
| orderings: log-space mean, per type T | 0.2373 | 0.7345 | 0.9630 | 0.0001 | 0.0001 |
| [diagnostic] per type T + histogram binning | 0.1955 | 0.7689 | 0.9559 | 0.0153 | -0.0008 |
| [diagnostic] per type T + isotonic (top label) | 0.2463 | 0.7264 | 0.9637 | 0.0267 | -0.0026 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: none (T = 1): 0/8, per source (upper bound): 2/429, [diagnostic] per type T + isotonic (top label): 4/1170

## generators: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4054 | 0.0948 | 0.9144 | 0.1726 | 0.5818 | 0.8376 | 0.0602 |
| global T | 0.3395 | 0.0779 | 0.7310 | 0.1634 | 0.4996 | 0.6494 | 0.0642 |
| per type (old artifact) | 0.3247 | 0.0739 | 0.7214 | 0.1650 | 0.4681 | 0.6547 | 0.0604 |
| per lang | 0.3453 | 0.0801 | 0.6783 | 0.1711 | 0.4932 | 0.6065 | 0.0647 |
| per type × lang, unshrunk | 0.3263 | 0.0744 | 0.6787 | 0.1712 | 0.4693 | 0.6100 | 0.0604 |
| per type × lang, ridge (CV tau) [impl] | 0.3263 | 0.0744 | 0.6826 | 0.1716 | 0.4690 | 0.6145 | 0.0604 |
| per type × lang, ridge, served (guessed lang) | 0.3263 | 0.0744 | 0.6825 | 0.1717 | 0.4690 | 0.6144 | 0.0604 |
| per type × lang, empirical Bayes | 0.3261 | 0.0744 | 0.6990 | 0.1732 | 0.4683 | 0.6323 | 0.0604 |
| T(n) = a + b ln n | 0.3380 | 0.0798 | 0.7510 | 0.1727 | 0.4819 | 0.6863 | 0.0653 |
| T(n) = a_type + b ln n | 0.3241 | 0.0739 | 0.7210 | 0.1649 | 0.4675 | 0.6542 | 0.0604 |
| per source (upper bound) | 0.2856 | 0.0445 | 0.5193 | 0.1134 | 0.4077 | 0.4443 | 0.1204 |
| feature T (thermometer-lite) | 0.3179 | 0.0737 | 0.6804 | 0.1717 | 0.4636 | 0.6075 | 0.0627 |
| vector scaling: T + position bias, choice | 0.3262 | 0.0743 | 0.7160 | 0.1663 | 0.4681 | 0.6506 | 0.0604 |
| vector scaling: T + position bias, every type | 0.3249 | 0.0740 | 0.7150 | 0.1681 | 0.4659 | 0.6506 | 0.0593 |
| batch calibration (eval batch), choice+noul | 0.3276 | 0.0744 | 0.7138 | 0.1645 | 0.4682 | 0.6497 | 0.0604 |
| batch calibration (calibration split), choice+noul | 0.3247 | 0.0740 | 0.7202 | 0.1648 | 0.4681 | 0.6533 | 0.0604 |
| per type T + Platt with yes-bias (noul) | 0.3222 | 0.0747 | 0.7244 | 0.1393 | 0.4683 | 0.6547 | 0.0604 |
| per type T + beta (noul) | 0.3219 | 0.0747 | 0.7243 | 0.1394 | 0.4680 | 0.6547 | 0.0604 |
| orderings: first only, per type T | 0.3220 | 0.0663 | 0.7128 | 0.1656 | 0.4687 | 0.6418 | 0.0604 |
| orderings: mean of probabilities, per type T | 0.3247 | 0.0739 | 0.7214 | 0.1650 | 0.4681 | 0.6547 | 0.0604 |
| orderings: log-space mean, per type T | 0.3245 | 0.0742 | 0.7199 | 0.1654 | 0.4683 | 0.6526 | 0.0604 |
| [diagnostic] per type T + histogram binning | 0.3303 | 0.0781 | 0.7271 | 0.1483 | 0.4677 | 0.6689 | 0.0532 |
| [diagnostic] per type T + isotonic (top label) | 0.3295 | 0.0747 | 0.7553 | 0.1538 | 0.4811 | 0.6818 | 0.0667 |

## generators: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| none (T = 1) | 0.2800 | 0.6300 | 0.8750 | 0.0000 | 0.0000 |
| global T | 0.3600 | 0.5900 | 0.9219 | 0.0000 | 0.0000 |
| per type (old artifact) | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| per lang | 0.3700 | 0.5800 | 0.9206 | 0.0000 | 0.0000 |
| per type × lang, unshrunk | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| per type × lang, ridge (CV tau) [impl] | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| per type × lang, ridge, served (guessed lang) | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| per type × lang, empirical Bayes | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| T(n) = a + b ln n | 0.3900 | 0.5600 | 0.9180 | 0.0000 | 0.0000 |
| T(n) = a_type + b ln n | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| per source (upper bound) | 0.5700 | 0.4200 | 0.9767 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, choice | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, every type | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| batch calibration (eval batch), choice+noul | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| batch calibration (calibration split), choice+noul | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| per type T + Platt with yes-bias (noul) | 0.4000 | 0.5500 | 0.9167 | 0.0273 | 0.0114 |
| per type T + beta (noul) | 0.4000 | 0.5500 | 0.9167 | 0.0273 | 0.0114 |
| orderings: first only, per type T | 0.3900 | 0.5600 | 0.9180 | 0.0182 | 0.0045 |
| orderings: mean of probabilities, per type T | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| orderings: log-space mean, per type T | 0.4000 | 0.5500 | 0.9167 | 0.0000 | 0.0000 |
| [diagnostic] per type T + histogram binning | 0.3500 | 0.6000 | 0.9231 | 0.0000 | 0.0000 |
| [diagnostic] per type T + isotonic (top label) | 0.3900 | 0.5600 | 0.9180 | 0.0409 | 0.0237 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: per source (upper bound): 0/56, [diagnostic] per type T + isotonic (top label): 1/62

## Oracle (fitted on the evaluated set)

| method | NLL | ECE | en NLL | fr NLL |
|---|---|---|---|---|
| validation type | 0.4269 | 0.0074 | 0.4605 | 0.3106 |
| validation type_lang | 0.4258 | 0.0078 | 0.4602 | 0.3070 |
| generators type | 0.4250 | 0.0725 | 0.3098 | 0.5488 |
| generators type_lang | 0.4180 | 0.0692 | 0.2966 | 0.5486 |

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
 "none (T = 1)": {},
 "global T": {
  "temperatures": {
   "0": 1.3507564482037115
  },
  "global": 1.3507564482037115
 },
 "per type (old artifact)": {
  "temperatures": {
   "choice": 1.3368245073290927,
   "noul": 1.595452226483344,
   "score": 1.0661315010692751
  },
  "tau": null,
  "cv_nll": null
 },
 "per lang": {
  "temperatures": {
   "en": 1.305402795423888,
   "fr": 1.5162468999616052
  },
  "global": 1.3507564482037115
 },
 "per type \u00d7 lang, unshrunk": {
  "temperatures": {
   "choice": 1.3368245073290927,
   "noul": 1.595452226483344,
   "score": 1.0661315010692751,
   "choice/en": 1.2842384339721722,
   "choice/fr": 1.507370790544327,
   "noul/en": 1.622182574026627,
   "noul/fr": 1.5417423095698604,
   "score/en": 1.067591086736454
  },
  "tau": 0.0,
  "cv_nll": null
 },
 "per type \u00d7 lang, ridge (CV tau) [impl]": {
  "temperatures": {
   "choice": 1.3368245073290927,
   "noul": 1.595452226483344,
   "score": 1.0661315010692751,
   "choice/en": 1.2861587102283216,
   "choice/fr": 1.4888479566163313,
   "noul/en": 1.6200982445256615,
   "noul/fr": 1.5494777356898757,
   "score/en": 1.0674857683887125
  },
  "tau": 30.0,
  "cv_nll": {
   "0.0": 0.42759642242094487,
   "10.0": 0.4275918193365194,
   "30.0": 0.42758802726203393,
   "100.0": 0.42760123365095515,
   "300.0": 0.42767023214962097,
   "1000.0": 0.42779272631305315,
   "3000.0": 0.42787958342119625,
   "inf": 0.42794991358314677
  }
 },
 "per type \u00d7 lang, ridge, served (guessed lang)": {
  "temperatures": {
   "choice": 1.3368245073290927,
   "noul": 1.595452226483344,
   "score": 1.0661315010692751,
   "choice/en": 1.2861587102283216,
   "choice/fr": 1.4888479566163313,
   "noul/en": 1.6200982445256615,
   "noul/fr": 1.5494777356898757,
   "score/en": 1.0674857683887125
  },
  "tau": 30.0,
  "cv_nll": {
   "0.0": 0.42759642242094487,
   "10.0": 0.4275918193365194,
   "30.0": 0.42758802726203393,
   "100.0": 0.42760123365095515,
   "300.0": 0.42767023214962097,
   "1000.0": 0.42779272631305315,
   "3000.0": 0.42787958342119625,
   "inf": 0.42794991358314677
  }
 },
 "per type \u00d7 lang, empirical Bayes": {
  "temperatures": {
   "choice": 1.3368245073290927,
   "noul": 1.595452226483344,
   "score": 1.0661315010692751,
   "choice/en": 1.2974685122936498,
   "choice/fr": 1.418657505637156,
   "noul/en": 1.610744512650704,
   "noul/fr": 1.5739408399068193,
   "score/en": 1.0669973104039512
  },
  "tau": 266.3159083832344,
  "cv_nll": null,
  "eb_tau": 266.3159083832344,
  "sigma_log_t": 0.04332977633734401
 },
 "T(n) = a + b ln n": {
  "a_b_atype": [
   1.5992459854417458,
   -0.17998303726593118
  ],
  "types": [
   "choice",
   "noul",
   "score"
  ]
 },
 "T(n) = a_type + b ln n": {
  "a_b_atype": [
   1.286284174432463,
   0.0276361851602666,
   0.29608460859053237,
   -0.2607448895760083
  ],
  "types": [
   "choice",
   "noul",
   "score"
  ]
 },
 "per source (upper bound)": {
  "temperatures": {
   "allocine_fr": 0.9826988374374412,
   "banking77": 0.9275884627776809,
   "boolq": 1.2329695924053377,
   "certo": 0.9419688028526536,
   "clinc_oos": 1.0029297661525893,
   "dbpedia": 1.1285625741578091,
   "decider_teacher": 1.456485447699657,
   "gen_base_rate": 1.9965893698263797,
   "gen_invoice_total": 4.10575584344078,
   "gen_long_refund_policy": 1.5974613249977219,
   "gen_refund_policy": 1.9833919181225228,
   "gen_register": 0.24936531429745684,
   "gen_return_window": 9.125145619205723,
   "gen_schedule_conflict": 0.11944811304723506,
   "gen_spelling_error": 2.6017384772166703,
   "gen_table_argmax": 0.3935607852540879,
   "gen_ticket_triage": 1.1282837889369965,
   "gen_weekday": 3.3044469456243,
   "go_emotions": 0.9313947608895083,
   "gsm8k_check": 1.6132452294443338,
   "hc3_fr": 0.9840851246920759,
   "helpsteer2_correctness": 1.0968644314357126,
   "helpsteer2_helpfulness": 1.0166840888883601,
   "jailbreak": 1.1284062261174073,
   "mage": 1.0678671419591375,
   "massive_en": 0.9989470896491713,
   "massive_fr": 1.1056005425324529,
   "mghafiri_scenarios": 1.1392468955000614,
   "mlma_en": 1.0284020797328115,
   "mlma_fr": 1.9464944129190414,
   "mnli": 1.1142866991429428,
   "openjev": 1.5609786098798601,
   "pavlick_formality": 1.090987136652537,
   "paws": 1.298753645603239,
   "pawsx_fr": 1.375771873352257,
   "plumb": 2.2290995424859457,
   "prompt_injection": 0.05000215213148425,
   "raid": 1.1015572319329578,
   "snli": 1.1992135421745822,
   "squad2": 1.1666281928304416,
   "sst5": 1.0321906672251997,
   "stsb": 1.1028973959707298,
   "textdetox_fr": 0.7236444589187193,
   "tweet_sentiment_en": 1.2648971912493707,
   "tweet_sentiment_fr": 1.4567711718212653,
   "vitaminc": 1.0252345894247736,
   "xnli_fr": 1.2735567498885065
  },
  "global": 1.3507564482037115
 },
 "feature T (thermometer-lite)": {
  "type_log_t": {
   "choice": 0.27194779810922454,
   "noul": 0.42854403863518686,
   "score": 0.1305056085528235
  },
  "w": [
   0.007133712977735331,
   0.01719907666498084,
   -0.07286393583143838,
   -0.06390552986698722,
   -0.037624658921061954,
   -0.0034804821178609454
  ],
  "features": [
   "ln_options",
   "lang_fr",
   "lang_unknown",
   "ln_tokens",
   "entropy",
   "max_prob"
  ],
  "lambda": 100.0,
  "cv_nll": {
   "0.1": 0.4266619934491251,
   "1.0": 0.42666105866315074,
   "10.0": 0.4266530070445004,
   "100.0": 0.4266161289595723,
   "1000.0": 0.42678605471147923,
   "10000.0": 0.4276278920956606,
   "inf": 0.4279499185238393
  }
 },
 "vector scaling: T + position bias, choice": {
  "choice": {
   "T": 1.3239905773117715,
   "bias": [
    0.245,
    0.227,
    0.102,
    -0.296,
    -0.569,
    -0.249,
    -0.05
   ]
  },
  "noul": {
   "T": 1.5954514509986677,
   "bias": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ]
  },
  "score": {
   "T": 1.0661340038573908,
   "bias": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ]
  }
 },
 "vector scaling: T + position bias, every type": {
  "choice": {
   "T": 1.3239905773117
```
