n: {'calibration': 11187, 'validation': 11233, 'generators': 220}


## All slots

| method | CV NLL (cal) | validation NLL | validation ECE | validation Brier | validation acc | generators NLL | generators ECE | generators Brier | generators acc |
|---|---|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4547 | 0.4519 | 0.0344 | 0.2175 | 0.8077 | 0.4044 | 0.0785 | 0.2059 | 0.8081 |
| global T | 0.4504 | 0.4474 | 0.0167 | 0.2150 | 0.8075 | 0.3906 | 0.0640 | 0.2011 | 0.8081 |
| per type (old artifact) | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.8076 | 0.3838 | 0.0545 | 0.1987 | 0.8081 |
| per lang | 0.4502 | 0.4471 | 0.0169 | 0.2150 | 0.8076 | 0.3890 | 0.0699 | 0.2002 | 0.8081 |
| per type × lang, unshrunk | 0.4491 | 0.4460 | 0.0174 | 0.2147 | 0.8076 | 0.3830 | 0.0663 | 0.1985 | 0.8081 |
| per type × lang, ridge (CV tau) [impl] | 0.4491 | 0.4460 | 0.0174 | 0.2147 | 0.8076 | 0.3830 | 0.0663 | 0.1985 | 0.8081 |
| per type × lang, ridge, served (guessed lang) | 0.4491 | 0.4462 | 0.0174 | 0.2147 | 0.8076 | 0.3829 | 0.0664 | 0.1985 | 0.8081 |
| per type × lang, empirical Bayes | 0.4492 | 0.4462 | 0.0179 | 0.2147 | 0.8076 | 0.3832 | 0.0541 | 0.1986 | 0.8081 |
| T(n) = a + b ln n | 0.4484 | 0.4469 | 0.0160 | 0.2146 | 0.8077 | 0.3846 | 0.0520 | 0.1987 | 0.8081 |
| T(n) = a_type + b ln n | 0.4487 | 0.4467 | 0.0177 | 0.2146 | 0.8077 | 0.3834 | 0.0502 | 0.1984 | 0.8081 |
| per source (upper bound) | 0.4434 | 0.4483 | 0.0151 | 0.2122 | 0.8078 | 0.3643 | 0.0285 | 0.1890 | 0.8081 |
| feature T (thermometer-lite) | 0.4474 | 0.4452 | 0.0121 | 0.2140 | 0.8077 | 0.3821 | 0.0415 | 0.1959 | 0.8081 |
| vector scaling: T + position bias, choice | 0.4496 | 0.4469 | 0.0186 | 0.2150 | 0.8064 | 0.3842 | 0.0595 | 0.1989 | 0.8036 |
| vector scaling: T + position bias, every type | 0.4464 | 0.4443 | 0.0121 | 0.2140 | 0.8073 | 0.3808 | 0.0476 | 0.1950 | 0.8036 |
| batch calibration (eval batch), choice+noul | 0.4504 | 0.4469 | 0.0180 | 0.2150 | 0.8073 | 0.3834 | 0.0568 | 0.1988 | 0.8172 |
| batch calibration (calibration split), choice+noul | 0.4491 | 0.4466 | 0.0172 | 0.2148 | 0.8074 | 0.3844 | 0.0546 | 0.1989 | 0.8081 |
| per type T + Platt with yes-bias (noul) | 0.4494 | 0.4463 | 0.0177 | 0.2147 | 0.8089 | 0.3851 | 0.0518 | 0.1994 | 0.8079 |
| per type T + beta (noul) | 0.4494 | 0.4464 | 0.0179 | 0.2147 | 0.8088 | 0.3850 | 0.0495 | 0.1995 | 0.8079 |
| orderings: first only, per type T | 0.4515 | 0.4489 | 0.0151 | 0.2158 | 0.8078 | 0.3823 | 0.0560 | 0.1981 | 0.8004 |
| orderings: mean of probabilities, per type T | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.8076 | 0.3838 | 0.0545 | 0.1987 | 0.8081 |
| orderings: log-space mean, per type T | 0.4495 | 0.4464 | 0.0176 | 0.2147 | 0.8077 | 0.3839 | 0.0510 | 0.1986 | 0.8081 |
| [diagnostic] per type T + histogram binning | 0.4527 | 0.4489 | 0.0112 | 0.2161 | 0.8060 | 0.3924 | 0.0523 | 0.2006 | 0.8081 |
| [diagnostic] per type T + isotonic (top label) | 0.4573 | 0.4481 | 0.0089 | 0.2140 | 0.8084 | 0.3749 | 0.0570 | 0.1916 | 0.8177 |

## validation: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4925 | 0.0342 | 0.3113 | 0.0361 | 0.2796 | 0.4287 | 0.9634 |
| global T | 0.4890 | 0.0154 | 0.3033 | 0.0239 | 0.2723 | 0.4248 | 0.9645 |
| per type (old artifact) | 0.4877 | 0.0179 | 0.3042 | 0.0245 | 0.2711 | 0.4248 | 0.9624 |
| per lang | 0.4890 | 0.0182 | 0.3023 | 0.0211 | 0.2728 | 0.4239 | 0.9639 |
| per type × lang, unshrunk | 0.4876 | 0.0181 | 0.3020 | 0.0194 | 0.2710 | 0.4237 | 0.9624 |
| per type × lang, ridge (CV tau) [impl] | 0.4876 | 0.0181 | 0.3021 | 0.0202 | 0.2710 | 0.4237 | 0.9624 |
| per type × lang, ridge, served (guessed lang) | 0.4877 | 0.0179 | 0.3027 | 0.0221 | 0.2709 | 0.4242 | 0.9624 |
| per type × lang, empirical Bayes | 0.4876 | 0.0176 | 0.3028 | 0.0222 | 0.2711 | 0.4240 | 0.9624 |
| T(n) = a + b ln n | 0.4884 | 0.0160 | 0.3034 | 0.0194 | 0.2712 | 0.4254 | 0.9632 |
| T(n) = a_type + b ln n | 0.4878 | 0.0171 | 0.3046 | 0.0216 | 0.2713 | 0.4248 | 0.9631 |
| per source (upper bound) | 0.4938 | 0.0193 | 0.2908 | 0.0097 | 0.2749 | 0.4240 | 0.9655 |
| feature T (thermometer-lite) | 0.4871 | 0.0126 | 0.3004 | 0.0158 | 0.2686 | 0.4241 | 0.9625 |
| vector scaling: T + position bias, choice | 0.4879 | 0.0175 | 0.3052 | 0.0237 | 0.2711 | 0.4257 | 0.9624 |
| vector scaling: T + position bias, every type | 0.4852 | 0.0121 | 0.3029 | 0.0167 | 0.2684 | 0.4257 | 0.9534 |
| batch calibration (eval batch), choice+noul | 0.4881 | 0.0170 | 0.3045 | 0.0242 | 0.2712 | 0.4256 | 0.9624 |
| batch calibration (calibration split), choice+noul | 0.4877 | 0.0161 | 0.3045 | 0.0254 | 0.2712 | 0.4249 | 0.9624 |
| per type T + Platt with yes-bias (noul) | 0.4877 | 0.0177 | 0.3033 | 0.0197 | 0.2707 | 0.4248 | 0.9624 |
| per type T + beta (noul) | 0.4877 | 0.0181 | 0.3034 | 0.0195 | 0.2707 | 0.4248 | 0.9624 |
| orderings: first only, per type T | 0.4898 | 0.0147 | 0.3074 | 0.0193 | 0.2721 | 0.4294 | 0.9624 |
| orderings: mean of probabilities, per type T | 0.4877 | 0.0179 | 0.3042 | 0.0245 | 0.2711 | 0.4248 | 0.9624 |
| orderings: log-space mean, per type T | 0.4875 | 0.0172 | 0.3043 | 0.0231 | 0.2712 | 0.4243 | 0.9624 |
| [diagnostic] per type T + histogram binning | 0.4901 | 0.0133 | 0.3063 | 0.0155 | 0.2711 | 0.4289 | 0.9662 |
| [diagnostic] per type T + isotonic (top label) | 0.4898 | 0.0106 | 0.3038 | 0.0170 | 0.2697 | 0.4248 | 0.9758 |

## validation: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| none (T = 1) | 0.1718 | 0.7818 | 0.9439 | 0.0000 | 0.0000 |
| global T | 0.2069 | 0.7527 | 0.9490 | 0.0002 | -0.0002 |
| per type (old artifact) | 0.2373 | 0.7287 | 0.9554 | 0.0001 | -0.0001 |
| per lang | 0.2075 | 0.7516 | 0.9484 | 0.0001 | -0.0001 |
| per type × lang, unshrunk | 0.2377 | 0.7287 | 0.9559 | 0.0001 | -0.0001 |
| per type × lang, ridge (CV tau) [impl] | 0.2379 | 0.7285 | 0.9559 | 0.0001 | -0.0001 |
| per type × lang, ridge, served (guessed lang) | 0.2368 | 0.7294 | 0.9557 | 0.0001 | -0.0001 |
| per type × lang, empirical Bayes | 0.2375 | 0.7283 | 0.9551 | 0.0001 | -0.0001 |
| T(n) = a + b ln n | 0.2388 | 0.7274 | 0.9556 | 0.0000 | 0.0000 |
| T(n) = a_type + b ln n | 0.2433 | 0.7242 | 0.9570 | 0.0000 | 0.0000 |
| per source (upper bound) | 0.2279 | 0.7388 | 0.9568 | 0.0002 | 0.0001 |
| feature T (thermometer-lite) | 0.2724 | 0.7011 | 0.9635 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, choice | 0.2373 | 0.7287 | 0.9554 | 0.0134 | -0.0013 |
| vector scaling: T + position bias, every type | 0.2698 | 0.7033 | 0.9631 | 0.0318 | -0.0003 |
| batch calibration (eval batch), choice+noul | 0.2373 | 0.7287 | 0.9554 | 0.0052 | -0.0003 |
| batch calibration (calibration split), choice+noul | 0.2373 | 0.7287 | 0.9554 | 0.0059 | -0.0003 |
| per type T + Platt with yes-bias (noul) | 0.2300 | 0.7343 | 0.9536 | 0.0028 | 0.0012 |
| per type T + beta (noul) | 0.2302 | 0.7341 | 0.9536 | 0.0034 | 0.0012 |
| orderings: first only, per type T | 0.2386 | 0.7281 | 0.9562 | 0.0161 | 0.0002 |
| orderings: mean of probabilities, per type T | 0.2373 | 0.7287 | 0.9554 | 0.0001 | -0.0001 |
| orderings: log-space mean, per type T | 0.2373 | 0.7283 | 0.9548 | 0.0007 | 0.0000 |
| [diagnostic] per type T + histogram binning | 0.3019 | 0.6763 | 0.9687 | 0.0171 | -0.0017 |
| [diagnostic] per type T + isotonic (top label) | 0.2373 | 0.7287 | 0.9554 | 0.0194 | 0.0007 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: per source (upper bound): 5/470, [diagnostic] per type T + isotonic (top label): 5/540

## generators: by language and type

| method | lang=en nll | lang=en ece | lang=fr nll | lang=fr ece | type=noul NLL | type=choice NLL | type=score NLL |
|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.3203 | 0.0527 | 0.4950 | 0.1141 | 0.4784 | 0.4070 | 0.0217 |
| global T | 0.3066 | 0.0460 | 0.4809 | 0.0983 | 0.4514 | 0.4026 | 0.0266 |
| per type (old artifact) | 0.2978 | 0.0468 | 0.4763 | 0.0876 | 0.4367 | 0.4030 | 0.0234 |
| per lang | 0.3081 | 0.0474 | 0.4760 | 0.1066 | 0.4493 | 0.4012 | 0.0267 |
| per type × lang, unshrunk | 0.2966 | 0.0470 | 0.4759 | 0.1069 | 0.4371 | 0.4008 | 0.0234 |
| per type × lang, ridge (CV tau) [impl] | 0.2967 | 0.0470 | 0.4759 | 0.1069 | 0.4371 | 0.4009 | 0.0234 |
| per type × lang, ridge, served (guessed lang) | 0.2966 | 0.0470 | 0.4757 | 0.1071 | 0.4370 | 0.4007 | 0.0234 |
| per type × lang, empirical Bayes | 0.2970 | 0.0470 | 0.4758 | 0.0905 | 0.4366 | 0.4017 | 0.0234 |
| T(n) = a + b ln n | 0.2961 | 0.0448 | 0.4799 | 0.0758 | 0.4358 | 0.4047 | 0.0284 |
| T(n) = a_type + b ln n | 0.2956 | 0.0451 | 0.4779 | 0.0759 | 0.4339 | 0.4042 | 0.0267 |
| per source (upper bound) | 0.2781 | 0.0365 | 0.4571 | 0.0466 | 0.3981 | 0.4035 | 0.0000 |
| feature T (thermometer-lite) | 0.2953 | 0.0373 | 0.4754 | 0.0686 | 0.4330 | 0.4020 | 0.0274 |
| vector scaling: T + position bias, choice | 0.2965 | 0.0549 | 0.4786 | 0.0870 | 0.4367 | 0.4039 | 0.0234 |
| vector scaling: T + position bias, every type | 0.2953 | 0.0450 | 0.4727 | 0.0887 | 0.4295 | 0.4039 | 0.0209 |
| batch calibration (eval batch), choice+noul | 0.2988 | 0.0493 | 0.4744 | 0.0891 | 0.4368 | 0.4019 | 0.0234 |
| batch calibration (calibration split), choice+noul | 0.2981 | 0.0540 | 0.4771 | 0.0811 | 0.4367 | 0.4042 | 0.0234 |
| per type T + Platt with yes-bias (noul) | 0.2967 | 0.0420 | 0.4800 | 0.0790 | 0.4394 | 0.4030 | 0.0234 |
| per type T + beta (noul) | 0.2965 | 0.0420 | 0.4802 | 0.0793 | 0.4393 | 0.4030 | 0.0234 |
| orderings: first only, per type T | 0.2917 | 0.0464 | 0.4798 | 0.0797 | 0.4417 | 0.3947 | 0.0234 |
| orderings: mean of probabilities, per type T | 0.2978 | 0.0468 | 0.4763 | 0.0876 | 0.4367 | 0.4030 | 0.0234 |
| orderings: log-space mean, per type T | 0.2978 | 0.0468 | 0.4765 | 0.0795 | 0.4371 | 0.4028 | 0.0234 |
| [diagnostic] per type T + histogram binning | 0.3030 | 0.0399 | 0.4886 | 0.0780 | 0.4339 | 0.4238 | 0.0281 |
| [diagnostic] per type T + isotonic (top label) | 0.2902 | 0.0334 | 0.4660 | 0.0991 | 0.4267 | 0.3934 | 0.0235 |

## generators: noul in the v1.5 band [0.2, 0.8], and flags

| method | noul abstain | noul v1.5 acc | noul committed acc | argmax moved | Δ accuracy |
|---|---|---|---|---|---|
| none (T = 1) | 0.3500 | 0.6000 | 0.9231 | 0.0000 | 0.0000 |
| global T | 0.3900 | 0.5700 | 0.9344 | 0.0000 | 0.0000 |
| per type (old artifact) | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| per lang | 0.4100 | 0.5600 | 0.9492 | 0.0000 | 0.0000 |
| per type × lang, unshrunk | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| per type × lang, ridge (CV tau) [impl] | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| per type × lang, ridge, served (guessed lang) | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| per type × lang, empirical Bayes | 0.4300 | 0.5400 | 0.9474 | 0.0000 | 0.0000 |
| T(n) = a + b ln n | 0.4300 | 0.5400 | 0.9474 | 0.0000 | 0.0000 |
| T(n) = a_type + b ln n | 0.4400 | 0.5300 | 0.9464 | 0.0000 | 0.0000 |
| per source (upper bound) | 0.5500 | 0.4400 | 0.9778 | 0.0000 | 0.0000 |
| feature T (thermometer-lite) | 0.4400 | 0.5300 | 0.9464 | 0.0000 | 0.0000 |
| vector scaling: T + position bias, choice | 0.4200 | 0.5500 | 0.9483 | 0.0636 | -0.0045 |
| vector scaling: T + position bias, every type | 0.4400 | 0.5300 | 0.9464 | 0.0636 | -0.0045 |
| batch calibration (eval batch), choice+noul | 0.4200 | 0.5500 | 0.9483 | 0.0227 | 0.0091 |
| batch calibration (calibration split), choice+noul | 0.4200 | 0.5500 | 0.9483 | 0.0364 | 0.0000 |
| per type T + Platt with yes-bias (noul) | 0.4200 | 0.5400 | 0.9310 | 0.0091 | -0.0002 |
| per type T + beta (noul) | 0.4200 | 0.5400 | 0.9310 | 0.0091 | -0.0002 |
| orderings: first only, per type T | 0.4300 | 0.5300 | 0.9298 | 0.0500 | -0.0077 |
| orderings: mean of probabilities, per type T | 0.4200 | 0.5500 | 0.9483 | 0.0000 | 0.0000 |
| orderings: log-space mean, per type T | 0.4300 | 0.5400 | 0.9474 | 0.0000 | 0.0000 |
| [diagnostic] per type T + histogram binning | 0.4400 | 0.5300 | 0.9464 | 0.0000 | 0.0000 |
| [diagnostic] per type T + isotonic (top label) | 0.4200 | 0.5500 | 0.9483 | 0.0318 | 0.0096 |

Items with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: per source (upper bound): 0/59, [diagnostic] per type T + isotonic (top label): 0/38

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
 "none (T = 1)": {},
 "global T": {
  "temperatures": {
   "0": 1.1704267010740912
  },
  "global": 1.1704267010740912
 },
 "per type (old artifact)": {
  "temperatures": {
   "choice": 1.1332639044221633,
   "noul": 1.3120353701728706,
   "score": 1.062142349167514
  },
  "tau": null,
  "cv_nll": null
 },
 "per lang": {
  "temperatures": {
   "en": 1.1456948903815274,
   "fr": 1.2712293453186196
  },
  "global": 1.1704267010740912
 },
 "per type \u00d7 lang, unshrunk": {
  "temperatures": {
   "choice": 1.1332639044221633,
   "noul": 1.3120353701728706,
   "score": 1.062142349167514,
   "choice/en": 1.0933199615235405,
   "choice/fr": 1.2837356152976578,
   "noul/en": 1.334215753993507,
   "noul/fr": 1.2639013691970338,
   "score/en": 1.063822024326958
  },
  "tau": 0.0,
  "cv_nll": null
 },
 "per type \u00d7 lang, ridge (CV tau) [impl]": {
  "temperatures": {
   "choice": 1.1332639044221633,
   "noul": 1.3120353701728706,
   "score": 1.062142349167514,
   "choice/en": 1.093837089991389,
   "choice/fr": 1.2767942719204914,
   "noul/en": 1.3336806026826449,
   "noul/fr": 1.2663122016871078,
   "score/en": 1.0637776679346136
  },
  "tau": 10.0,
  "cv_nll": {
   "0.0": 0.44895966259263975,
   "10.0": 0.4489586097160088,
   "30.0": 0.4489614976864819,
   "100.0": 0.44899207865693563,
   "300.0": 0.4490787092068657,
   "1000.0": 0.4492021009615302,
   "3000.0": 0.4492802865158303,
   "inf": 0.4493400821153198
  }
 },
 "per type \u00d7 lang, ridge, served (guessed lang)": {
  "temperatures": {
   "choice": 1.1332639044221633,
   "noul": 1.3120353701728706,
   "score": 1.062142349167514,
   "choice/en": 1.093837089991389,
   "choice/fr": 1.2767942719204914,
   "noul/en": 1.3336806026826449,
   "noul/fr": 1.2663122016871078,
   "score/en": 1.0637776679346136
  },
  "tau": 10.0,
  "cv_nll": {
   "0.0": 0.44895966259263975,
   "10.0": 0.4489586097160088,
   "30.0": 0.4489614976864819,
   "100.0": 0.44899207865693563,
   "300.0": 0.4490787092068657,
   "1000.0": 0.4492021009615302,
   "3000.0": 0.4492802865158303,
   "inf": 0.4493400821153198
  }
 },
 "per type \u00d7 lang, empirical Bayes": {
  "temperatures": {
   "choice": 1.1332639044221633,
   "noul": 1.3120353701728706,
   "score": 1.062142349167514,
   "choice/en": 1.102873225062419,
   "choice/fr": 1.2027470976845762,
   "noul/en": 1.3260393671190227,
   "noul/fr": 1.2907490773900678,
   "score/en": 1.0632005111241076
  },
  "tau": 238.83905489259186,
  "cv_nll": null,
  "eb_tau": 238.83905489259186,
  "sigma_log_t": 0.045754343696635835
 },
 "T(n) = a + b ln n": {
  "a_b_atype": [
   1.4876178535880662,
   -0.23742657881918672
  ],
  "types": [
   "choice",
   "noul",
   "score"
  ]
 },
 "T(n) = a_type + b ln n": {
  "a_b_atype": [
   1.276204525950429,
   -0.1153965687893655,
   0.14963958077406248,
   0.025438998409908722
  ],
  "types": [
   "choice",
   "noul",
   "score"
  ]
 },
 "per source (upper bound)": {
  "temperatures": {
   "allocine_fr": 0.8291284538072076,
   "banking77": 1.06286804915991,
   "boolq": 1.248843812054185,
   "certo": 0.9358274509792878,
   "clinc_oos": 1.041709570586603,
   "dbpedia": 0.8869839815561111,
   "decider_teacher": 1.0845804646683268,
   "gen_base_rate": 1.7486128184396492,
   "gen_invoice_total": 5.77669744259676,
   "gen_long_refund_policy": 1.3824881674435363,
   "gen_refund_policy": 1.291594316637276,
   "gen_register": 0.050002504231074246,
   "gen_return_window": 4.407195346101117,
   "gen_schedule_conflict": 0.06040013370871818,
   "gen_spelling_error": 2.095066065775765,
   "gen_table_argmax": 0.05000215410055939,
   "gen_ticket_triage": 1.015626987089463,
   "gen_weekday": 0.797817527698079,
   "go_emotions": 0.9992438534752202,
   "gsm8k_check": 1.0217483530574532,
   "hc3_fr": 0.8213328318641413,
   "helpsteer2_correctness": 1.1276729182382663,
   "helpsteer2_helpfulness": 0.7883084968002566,
   "jailbreak": 1.3331754147178303,
   "mage": 0.9882500328552226,
   "massive_en": 0.9070203120670889,
   "massive_fr": 0.9584246006779995,
   "mghafiri_scenarios": 1.2470452952695807,
   "mlma_en": 0.8408080027928547,
   "mlma_fr": 1.5628598597599839,
   "mnli": 1.1620575936217084,
   "openjev": 1.1814452154091215,
   "pavlick_formality": 1.216100213866824,
   "paws": 1.2885813257653846,
   "pawsx_fr": 1.2676159568425829,
   "plumb": 1.0445137194206626,
   "prompt_injection": 0.08287621941235845,
   "raid": 1.0914670582613573,
   "snli": 1.3131166228726558,
   "squad2": 1.2342672853350953,
   "sst5": 0.9589176723908343,
   "stsb": 1.1136996620591255,
   "textdetox_fr": 0.5795652365226655,
   "tweet_sentiment_en": 1.2363680431615756,
   "tweet_sentiment_fr": 1.654869449255856,
   "vitaminc": 1.2182148876053622,
   "xnli_fr": 1.2939303300162708
  },
  "global": 1.1704267010740912
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
 "vector scaling: T + position bias, choice": {
  "choice": {
   "T": 1.1254423232987265,
   "bias": [
    0.127,
    0.334,
    0.588,
    0.194,
    -0.082,
    -0.303,
    -0.182
   ]
  },
  "noul": {
   "T": 1.3120158886647377,
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
   "T": 1.0621342448203435,
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
   "T": 1.1254423232987265,
   
```
