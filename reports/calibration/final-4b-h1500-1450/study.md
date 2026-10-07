n: {'calibration': 11187, 'validation': 11233, 'generators': 220}


## All slots

| method | CV NLL (cal) | validation NLL | validation ECE | validation Brier | validation acc | generators NLL | generators ECE | generators Brier | generators acc |
|---|---|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4357 | 0.4338 | 0.0242 | 0.2078 | 0.8170 | 0.4058 | 0.0696 | 0.1923 | 0.8024 |
| global T | 0.4335 | 0.4315 | 0.0139 | 0.2067 | 0.8170 | 0.3956 | 0.0533 | 0.1902 | 0.8024 |
| per type (old artifact) | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.8170 | 0.3892 | 0.0585 | 0.1886 | 0.8024 |
| per lang | 0.4334 | 0.4312 | 0.0132 | 0.2067 | 0.8170 | 0.3961 | 0.0575 | 0.1897 | 0.8024 |
| per type × lang, unshrunk | 0.4320 | 0.4295 | 0.0138 | 0.2064 | 0.8170 | 0.3902 | 0.0556 | 0.1886 | 0.8024 |
| per type × lang, ridge (CV tau) [impl] | 0.4319 | 0.4296 | 0.0144 | 0.2064 | 0.8170 | 0.3897 | 0.0559 | 0.1886 | 0.8024 |
| per type × lang, ridge, served (guessed lang) | 0.4319 | 0.4296 | 0.0141 | 0.2064 | 0.8170 | 0.3897 | 0.0560 | 0.1886 | 0.8024 |
| per type × lang, empirical Bayes | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.8170 | 0.3892 | 0.0585 | 0.1886 | 0.8024 |
| T(n) = a + b ln n | 0.4315 | 0.4304 | 0.0106 | 0.2063 | 0.8170 | 0.3873 | 0.0646 | 0.1887 | 0.8024 |
| T(n) = a_type + b ln n | 0.4315 | 0.4300 | 0.0130 | 0.2063 | 0.8170 | 0.3867 | 0.0657 | 0.1885 | 0.8024 |
| per source (upper bound) | 0.4288 | 0.4306 | 0.0108 | 0.2042 | 0.8170 | 0.3655 | 0.0382 | 0.1868 | 0.8024 |
| feature T (thermometer-lite) | 0.4317 | 0.4295 | 0.0090 | 0.2060 | 0.8170 | 0.3862 | 0.0595 | 0.1877 | 0.8024 |
| vector scaling: T + position bias, choice | 0.4322 | 0.4301 | 0.0135 | 0.2065 | 0.8177 | 0.3874 | 0.0498 | 0.1880 | 0.8115 |
| vector scaling: T + position bias, every type | 0.4322 | 0.4301 | 0.0106 | 0.2065 | 0.8177 | 0.3879 | 0.0510 | 0.1873 | 0.8115 |
| batch calibration (eval batch), choice+noul | 0.4331 | 0.4301 | 0.0141 | 0.2066 | 0.8168 | 0.3851 | 0.0546 | 0.1874 | 0.8115 |
| batch calibration (calibration split), choice+noul | 0.4319 | 0.4299 | 0.0151 | 0.2064 | 0.8173 | 0.3892 | 0.0586 | 0.1885 | 0.8024 |
| per type T + Platt with yes-bias (noul) | 0.4320 | 0.4297 | 0.0139 | 0.2063 | 0.8174 | 0.3896 | 0.0576 | 0.1887 | 0.8060 |
| per type T + beta (noul) | 0.4318 | 0.4296 | 0.0132 | 0.2063 | 0.8176 | 0.3873 | 0.0611 | 0.1889 | 0.8056 |
| orderings: first only, per type T | 0.4336 | 0.4314 | 0.0107 | 0.2070 | 0.8169 | 0.3897 | 0.0576 | 0.1893 | 0.8115 |
| orderings: mean of probabilities, per type T | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.8170 | 0.3892 | 0.0585 | 0.1886 | 0.8024 |
| orderings: log-space mean, per type T | 0.4318 | 0.4297 | 0.0141 | 0.2063 | 0.8169 | 0.3889 | 0.0578 | 0.1884 | 0.8024 |
| [diagnostic] per type T + histogram binning | 0.4362 | 0.4346 | 0.0120 | 0.2093 | 0.8141 | 0.4037 | 0.0742 | 0.1989 | 0.8024 |
| [diagnostic] per type T + isotonic (top label) | 0.4417 | 0.4348 | 0.0086 | 0.2067 | 0.8178 | 0.3844 | 0.0509 | 0.1877 | 0.8101 |

## validation: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4726 | 0.0248 | 0.2996 | 0.0254 | 0.2797 | 0.3983 | 0.9305 |
| global T | 0.4711 | 0.0156 | 0.2944 | 0.0148 | 0.2736 | 0.3982 | 0.9325 |
| per type (old artifact) | 0.4689 | 0.0147 | 0.2943 | 0.0174 | 0.2708 | 0.3976 | 0.9304 |
| per lang | 0.4711 | 0.0158 | 0.2932 | 0.0083 | 0.2737 | 0.3976 | 0.9319 |
| per type × lang, unshrunk | 0.4688 | 0.0150 | 0.2933 | 0.0138 | 0.2708 | 0.3970 | 0.9304 |
| per type × lang, ridge (CV tau) [impl] | 0.4689 | 0.0152 | 0.2939 | 0.0169 | 0.2708 | 0.3973 | 0.9304 |
| per type × lang, ridge, served (guessed lang) | 0.4689 | 0.0149 | 0.2940 | 0.0165 | 0.2708 | 0.3974 | 0.9304 |
| per type × lang, empirical Bayes | 0.4689 | 0.0147 | 0.2943 | 0.0174 | 0.2708 | 0.3976 | 0.9304 |
| T(n) = a + b ln n | 0.4700 | 0.0124 | 0.2934 | 0.0117 | 0.2708 | 0.3990 | 0.9308 |
| T(n) = a_type + b ln n | 0.4695 | 0.0150 | 0.2935 | 0.0148 | 0.2708 | 0.3982 | 0.9305 |
| per source (upper bound) | 0.4734 | 0.0154 | 0.2826 | 0.0087 | 0.2693 | 0.4009 | 0.9308 |
| feature T (thermometer-lite) | 0.4690 | 0.0115 | 0.2929 | 0.0107 | 0.2696 | 0.3982 | 0.9303 |
| vector scaling: T + position bias, choice | 0.4692 | 0.0128 | 0.2949 | 0.0208 | 0.2708 | 0.3984 | 0.9304 |
| vector scaling: T + position bias, every type | 0.4694 | 0.0098 | 0.2945 | 0.0180 | 0.2704 | 0.3984 | 0.9317 |
| batch calibration (eval batch), choice+noul | 0.4693 | 0.0143 | 0.2942 | 0.0177 | 0.2708 | 0.3984 | 0.9304 |
| batch calibration (calibration split), choice+noul | 0.4690 | 0.0155 | 0.2944 | 0.0186 | 0.2708 | 0.3979 | 0.9304 |
| per type T + Platt with yes-bias (noul) | 0.4690 | 0.0147 | 0.2940 | 0.0148 | 0.2707 | 0.3976 | 0.9304 |
| per type T + beta (noul) | 0.4686 | 0.0139 | 0.2944 | 0.0139 | 0.2703 | 0.3976 | 0.9304 |
| orderings: first only, per type T | 0.4706 | 0.0133 | 0.2962 | 0.0172 | 0.2713 | 0.4011 | 0.9304 |
| orderings: mean of probabilities, per type T | 0.4689 | 0.0147 | 0.2943 | 0.0174 | 0.2708 | 0.3976 | 0.9304 |
| orderings: log-space mean, per type T | 0.4688 | 0.0147 | 0.2943 | 0.0171 | 0.2707 | 0.3975 | 0.9304 |
| [diagnostic] per type T + histogram binning | 0.4739 | 0.0121 | 0.2986 | 0.0137 | 0.2738 | 0.4038 | 0.9365 |
| [diagnostic] per type T + isotonic (top label) | 0.4756 | 0.0099 | 0.2940 | 0.0109 | 0.2702 | 0.4090 | 0.9337 |

## validation: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| none (T = 1) | 0.1804 | 0.7745 | 0.9449 | 0.0000 | 0.0000 |
| global T | 0.1966 | 0.7617 | 0.9481 | 0.0000 | 0.0000 |
| per type (old artifact) | 0.2272 | 0.7373 | 0.9540 | 0.0000 | 0.0000 |
| per lang | 0.1985 | 0.7599 | 0.9482 | 0.0000 | 0.0000 |
| per type × lang, unshrunk | 0.2274 | 0.7371 | 0.9540 | 0.0000 | 0.0000 |
| per type × lang, ridge (CV tau) [impl] | 0.2274 | 0.7371 | 0.9540 | 0.0000 | 0.0000 |
| per type × lang, ridge, served (guessed lang) | 0.2274 | 0.7371 | 0.9540 | 0.0000 | 0.0000 |
| per type × lang, empirical Bayes | 0.2272 | 0.7373 | 0.9540 | 0.0000 | 0.0000 |
| T(n) = a + b ln n | 0.2214 | 0.7420 | 0.9530 | 0.0000 | 0.0000 |
| T(n) = a_type + b ln n | 0.2272 | 0.7373 | 0.9540 | 0.0000 | 0.0000 |
| per source (upper bound) | 0.2321 | 0.7371 | 0.9599 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.2418 | 0.7268 | 0.9585 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, choice | 0.2272 | 0.7373 | 0.9540 | 0.0077 | 0.0008 |
| vector scaling: T + position bias, every type | 0.2332 | 0.7330 | 0.9559 | 0.0178 | 0.0008 |
| batch calibration (eval batch), choice+noul | 0.2272 | 0.7373 | 0.9540 | 0.0076 | -0.0001 |
| batch calibration (calibration split), choice+noul | 0.2272 | 0.7373 | 0.9540 | 0.0017 | 0.0004 |
| per type T + Platt with yes-bias (noul) | 0.2274 | 0.7373 | 0.9543 | 0.0015 | 0.0005 |
| per type T + beta (noul) | 0.2246 | 0.7394 | 0.9536 | 0.0047 | 0.0007 |
| orderings: first only, per type T | 0.2285 | 0.7368 | 0.9551 | 0.0127 | -0.0001 |
| orderings: mean of probabilities, per type T | 0.2272 | 0.7373 | 0.9540 | 0.0000 | 0.0000 |
| orderings: log-space mean, per type T | 0.2279 | 0.7366 | 0.9540 | 0.0003 | -0.0001 |
| [diagnostic] per type T + histogram binning | 0.2015 | 0.7578 | 0.9491 | 0.0284 | -0.0029 |
| [diagnostic] per type T + isotonic (top label) | 0.2067 | 0.7537 | 0.9501 | 0.0378 | 0.0008 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: per source (upper bound): 4/323, [diagnostic] per type T + isotonic (top label): 3/788

## generators: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.3879 | 0.0738 | 0.4251 | 0.0788 | 0.4708 | 0.4097 | 0.0620 |
| global T | 0.3723 | 0.0571 | 0.4207 | 0.0624 | 0.4543 | 0.4027 | 0.0664 |
| per type (old artifact) | 0.3659 | 0.0551 | 0.4143 | 0.0696 | 0.4381 | 0.4057 | 0.0625 |
| per lang | 0.3745 | 0.0581 | 0.4194 | 0.0599 | 0.4527 | 0.4054 | 0.0673 |
| per type × lang, unshrunk | 0.3669 | 0.0554 | 0.4152 | 0.0633 | 0.4381 | 0.4078 | 0.0625 |
| per type × lang, ridge (CV tau) [impl] | 0.3666 | 0.0554 | 0.4146 | 0.0642 | 0.4381 | 0.4069 | 0.0625 |
| per type × lang, ridge, served (guessed lang) | 0.3666 | 0.0553 | 0.4146 | 0.0642 | 0.4381 | 0.4068 | 0.0625 |
| per type × lang, empirical Bayes | 0.3659 | 0.0551 | 0.4143 | 0.0696 | 0.4381 | 0.4057 | 0.0625 |
| T(n) = a + b ln n | 0.3598 | 0.0549 | 0.4168 | 0.0826 | 0.4405 | 0.3977 | 0.0689 |
| T(n) = a_type + b ln n | 0.3599 | 0.0548 | 0.4154 | 0.0851 | 0.4381 | 0.3994 | 0.0658 |
| per source (upper bound) | 0.3349 | 0.0355 | 0.3985 | 0.0601 | 0.4039 | 0.3884 | 0.0592 |
| feature T (thermometer-lite) | 0.3616 | 0.0550 | 0.4127 | 0.0729 | 0.4355 | 0.4011 | 0.0650 |
| vector scaling: T + position bias, choice | 0.3643 | 0.0559 | 0.4121 | 0.0579 | 0.4381 | 0.4016 | 0.0625 |
| vector scaling: T + position bias, every type | 0.3655 | 0.0573 | 0.4119 | 0.0588 | 0.4399 | 0.4016 | 0.0589 |
| batch calibration (eval batch), choice+noul | 0.3654 | 0.0560 | 0.4062 | 0.0607 | 0.4381 | 0.3967 | 0.0625 |
| batch calibration (calibration split), choice+noul | 0.3660 | 0.0552 | 0.4141 | 0.0698 | 0.4381 | 0.4056 | 0.0625 |
| per type T + Platt with yes-bias (noul) | 0.3655 | 0.0550 | 0.4156 | 0.0779 | 0.4390 | 0.4057 | 0.0625 |
| per type T + beta (noul) | 0.3618 | 0.0612 | 0.4147 | 0.0726 | 0.4339 | 0.4057 | 0.0625 |
| orderings: first only, per type T | 0.3635 | 0.0626 | 0.4180 | 0.0683 | 0.4397 | 0.4052 | 0.0625 |
| orderings: mean of probabilities, per type T | 0.3659 | 0.0551 | 0.4143 | 0.0696 | 0.4381 | 0.4057 | 0.0625 |
| orderings: log-space mean, per type T | 0.3657 | 0.0553 | 0.4138 | 0.0701 | 0.4387 | 0.4044 | 0.0625 |
| [diagnostic] per type T + histogram binning | 0.3715 | 0.0631 | 0.4384 | 0.0860 | 0.4298 | 0.4364 | 0.1102 |
| [diagnostic] per type T + isotonic (top label) | 0.3607 | 0.0524 | 0.4098 | 0.0599 | 0.4316 | 0.4025 | 0.0577 |

## generators: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| none (T = 1) | 0.3700 | 0.5900 | 0.9365 | 0.0000 | 0.0000 |
| global T | 0.3800 | 0.5800 | 0.9355 | 0.0000 | 0.0000 |
| per type (old artifact) | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| per lang | 0.4000 | 0.5600 | 0.9333 | 0.0000 | 0.0000 |
| per type × lang, unshrunk | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| per type × lang, ridge (CV tau) [impl] | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| per type × lang, ridge, served (guessed lang) | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| per type × lang, empirical Bayes | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| T(n) = a + b ln n | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| T(n) = a_type + b ln n | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| per source (upper bound) | 0.6100 | 0.3900 | 1.0000 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.4400 | 0.5200 | 0.9286 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, choice | 0.4200 | 0.5400 | 0.9310 | 0.0273 | 0.0091 |
| vector scaling: T + position bias, every type | 0.4200 | 0.5400 | 0.9310 | 0.0273 | 0.0091 |
| batch calibration (eval batch), choice+noul | 0.4200 | 0.5400 | 0.9310 | 0.0545 | 0.0091 |
| batch calibration (calibration split), choice+noul | 0.4200 | 0.5400 | 0.9310 | 0.0045 | 0.0000 |
| per type T + Platt with yes-bias (noul) | 0.4200 | 0.5400 | 0.9310 | 0.0091 | 0.0035 |
| per type T + beta (noul) | 0.4200 | 0.5400 | 0.9310 | 0.0227 | 0.0031 |
| orderings: first only, per type T | 0.4100 | 0.5500 | 0.9322 | 0.0091 | 0.0091 |
| orderings: mean of probabilities, per type T | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| orderings: log-space mean, per type T | 0.4200 | 0.5400 | 0.9310 | 0.0000 | 0.0000 |
| [diagnostic] per type T + histogram binning | 0.3800 | 0.5800 | 0.9355 | 0.0000 | 0.0000 |
| [diagnostic] per type T + isotonic (top label) | 0.4000 | 0.5600 | 0.9333 | 0.0727 | 0.0077 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: per source (upper bound): 0/39, [diagnostic] per type T + isotonic (top label): 0/35

## Oracle (fitted on the evaluated set)

| method | NLL | ECE | en NLL | fr NLL |
|---|---|---|---|---|
| validation type | 0.4297 | 0.0143 | 0.4689 | 0.2943 |
| validation type_lang | 0.4294 | 0.0132 | 0.4688 | 0.2929 |
| generators type | 0.3812 | 0.0435 | 0.3474 | 0.4175 |
| generators type_lang | 0.3796 | 0.0560 | 0.3450 | 0.4168 |

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
   "?": 44,
   "en": 70
  },
  "fr": {
   "?": 17,
   "fr": 89
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
   "0": 1.1219925808644815
  },
  "global": 1.1219925808644815
 },
 "per type (old artifact)": {
  "temperatures": {
   "choice": 1.06000230239296,
   "noul": 1.3074502971678705,
   "score": 1.0146814391360806
  },
  "tau": null,
  "cv_nll": null
 },
 "per lang": {
  "temperatures": {
   "en": 1.1017050750114434,
   "fr": 1.2068278914374573
  },
  "global": 1.1219925808644815
 },
 "per type \u00d7 lang, unshrunk": {
  "temperatures": {
   "choice": 1.06000230239296,
   "noul": 1.3074502971678705,
   "score": 1.0146814391360806,
   "choice/en": 1.041894775215042,
   "choice/fr": 1.129152495135447,
   "noul/en": 1.3101549406577062,
   "noul/fr": 1.3014166716724291,
   "score/en": 1.015321839317105
  },
  "tau": 0.0,
  "cv_nll": null
 },
 "per type \u00d7 lang, ridge (CV tau) [impl]": {
  "temperatures": {
   "choice": 1.06000230239296,
   "noul": 1.3074502971678705,
   "score": 1.0146814391360806,
   "choice/en": 1.0473080837606694,
   "choice/fr": 1.0863489203478094,
   "noul/en": 1.3089845681079761,
   "noul/fr": 1.3052067879991476,
   "score/en": 1.0150475344948242
  },
  "tau": 300.0,
  "cv_nll": {
   "0.0": 0.43178558683537577,
   "10.0": 0.43177988547277724,
   "30.0": 0.4317715439534128,
   "100.0": 0.4317582116227051,
   "300.0": 0.4317527969620074,
   "1000.0": 0.4317583014873983,
   "3000.0": 0.4317653572688417,
   "inf": 0.43177208372107384
  }
 },
 "per type \u00d7 lang, ridge, served (guessed lang)": {
  "temperatures": {
   "choice": 1.06000230239296,
   "noul": 1.3074502971678705,
   "score": 1.0146814391360806,
   "choice/en": 1.0473080837606694,
   "choice/fr": 1.0863489203478094,
   "noul/en": 1.3089845681079761,
   "noul/fr": 1.3052067879991476,
   "score/en": 1.0150475344948242
  },
  "tau": 300.0,
  "cv_nll": {
   "0.0": 0.43178558683537577,
   "10.0": 0.43177988547277724,
   "30.0": 0.4317715439534128,
   "100.0": 0.4317582116227051,
   "300.0": 0.4317527969620074,
   "1000.0": 0.4317583014873983,
   "3000.0": 0.4317653572688417,
   "inf": 0.43177208372107384
  }
 },
 "per type \u00d7 lang, empirical Bayes": {
  "temperatures": {
   "choice": 1.06000230239296,
   "noul": 1.3074502971678705,
   "score": 1.0146814391360806,
   "choice/en": 1.0600017039254666,
   "choice/fr": 1.0600025395523198,
   "noul/en": 1.3074515079406033,
   "noul/fr": 1.3074505960626654,
   "score/en": 1.0146822544248881
  },
  "tau": 50000000.0,
  "cv_nll": null,
  "eb_tau": 50000000.0,
  "sigma_log_t": 0.0001
 },
 "T(n) = a + b ln n": {
  "a_b_atype": [
   1.4357702720647372,
   -0.23505497370910167
  ],
  "types": [
   "choice",
   "noul",
   "score"
  ]
 },
 "T(n) = a_type + b ln n": {
  "a_b_atype": [
   1.3434176453123166,
   -0.17738291702569586,
   0.0869966722204678,
   -0.041062937547378026
  ],
  "types": [
   "choice",
   "noul",
   "score"
  ]
 },
 "per source (upper bound)": {
  "temperatures": {
   "allocine_fr": 0.8694178304759426,
   "banking77": 0.9670204768755103,
   "boolq": 1.1404894772330898,
   "certo": 0.8869540109636044,
   "clinc_oos": 0.965526958044409,
   "dbpedia": 0.8122621693131409,
   "decider_teacher": 1.0822549463094395,
   "gen_base_rate": 1.825546698760907,
   "gen_invoice_total": 3.27705047494754,
   "gen_long_refund_policy": 1.3380933674283986,
   "gen_refund_policy": 1.3776035284395889,
   "gen_register": 0.6487229525494483,
   "gen_return_window": 5.548997146362383,
   "gen_schedule_conflict": 0.08409061917439765,
   "gen_spelling_error": 2.270011888796628,
   "gen_table_argmax": 0.0500440732901145,
   "gen_ticket_triage": 1.01341148437957,
   "gen_weekday": 0.8637578131793435,
   "go_emotions": 0.9790331081540713,
   "gsm8k_check": 1.0313685686134253,
   "hc3_fr": 1.047240791516599,
   "helpsteer2_correctness": 1.1065908570488765,
   "helpsteer2_helpfulness": 0.8963856153722938,
   "jailbreak": 1.0540578586261173,
   "mage": 1.0692193434441317,
   "massive_en": 0.9001986210495315,
   "massive_fr": 0.9025774545620977,
   "mghafiri_scenarios": 1.1244093140249978,
   "mlma_en": 1.0463719320256146,
   "mlma_fr": 1.8862687402716118,
   "mnli": 0.9885711165709553,
   "openjev": 1.101182452293092,
   "pavlick_formality": 1.02792349000213,
   "paws": 1.2053020939823418,
   "pawsx_fr": 1.1375828400889507,
   "plumb": 1.3274524360591722,
   "prompt_injection": 0.053576789384846696,
   "raid": 1.0428920814852327,
   "snli": 1.1598045776558281,
   "squad2": 1.155377168257985,
   "sst5": 0.9804535572884854,
   "stsb": 1.0005506337276362,
   "textdetox_fr": 0.6602784526645908,
   "tweet_sentiment_en": 1.1781285406833297,
   "tweet_sentiment_fr": 1.3482688279854484,
   "vitaminc": 1.0878362823246048,
   "xnli_fr": 1.0594353663711245
  },
  "global": 1.1219925808644815
 },
 "feature T (thermometer-lite)": {
  "type_log_t": {
   "choice": 0.10397829397251616,
   "noul": 0.1776354827971117,
   "score": 0.07361773045743485
  },
  "w": [
   -0.06537021268877287,
   -0.004749662125217117,
   -0.0397079429599578,
   -0.032381652309959104,
   0.09307728744473988,
   0.07009878167944754
  ],
  "features": [
   "ln_options",
   "lang_fr",
   "lang_unknown",
   "ln_tokens",
   "entropy",
   "max_prob"
  ],
  "lambda": 10.0,
  "cv_nll": {
   "0.1": 0.4310672472591604,
   "1.0": 0.4310666052107514,
   "10.0": 0.431061526015272,
   "100.0": 0.43106363852713353,
   "1000.0": 0.43124843601738616,
   "10000.0": 0.431646221397689,
   "inf": 0.43177213287912825
  }
 },
 "vector scaling: T + position bias, choice": {
  "choice": {
   "T": 1.0499512749442061,
   "bias": [
    0.372,
    0.43,
    0.486,
    0.272,
    -0.024,
    -0.137,
    -0.169
   ]
  },
  "noul": {
   "T": 1.3074294305062846,
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
   "T": 1.014680701600416,
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
   "T": 1.0499512749442061,
   "bias": [
    0.372,
    0.43,
  
```
