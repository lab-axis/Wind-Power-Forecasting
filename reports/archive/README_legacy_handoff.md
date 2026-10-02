# 상명풍력 시간별 발전량 예측 연구

> 이 문서는 이전 장흥 SCADA 기반 발전출력 예측 계획을 대체하는 최신 handoff README입니다.  
> 현재 연구의 핵심 목표는 **상명풍력발전소의 시간별 풍력 발전량을 기상관측 정보와 결합하여 예측하는 것**입니다.  
> Codex/다른 워크스페이스에서는 이 README를 기준으로 데이터 정제, 기상자료 병합, 시계열 전처리, EDA, 모델링을 진행합니다.

---

## 1. 연구 방향 변경 요약

초기에는 장흥풍력 SCADA의 `계통 유효 전력(kW)`을 이용한 발전출력 예측을 고려했습니다.

그러나 최종적으로 연구 관심을:

> **풍력 발전출력(Power)이 아니라 풍력 발전량(Energy) 예측**

으로 확정했습니다.

따라서 장흥 SCADA 대신 **상명풍력의 시간별 발전실적**을 주 데이터셋으로 사용합니다.

핵심 차이:

```text
장흥 SCADA
Target = 계통 유효 전력(kW)
=> 발전출력 예측

상명풍력
Target = 시간별 발전 에너지
=> 발전량 예측
```

본 연구에서는 상명풍력을 사용합니다.

---

# 2. 연구 목적

본 연구의 주요 질문은 다음과 같습니다.

1. 과거 풍력 발전량으로 미래 발전량을 얼마나 정확하게 예측할 수 있는가?
2. 인접 기상관측 정보가 발전량 예측 정확도를 향상시키는가?
3. 풍속, 풍향, 기온, 습도, 기압 등 어떤 기상변수가 발전량 예측에 가장 크게 기여하는가?
4. 기상정보의 시간적 지연(lag)이 발전량 예측에 의미가 있는가?
5. 예측 horizon이 길어질수록 모델 성능이 어떻게 변화하는가?
6. 단순 통계/머신러닝 모델과 딥러닝 시계열 모델의 성능 차이는 어떻게 나타나는가?
7. 더 가까운 기상관측소를 사용할 경우 예측 성능이 개선되는가?

---

# 3. 선행연구 연결

## 3.1 국내 KCI

### 박래진·강성우·이재형·정승민 (2022)
「정확도 향상을 위한 CNN-LSTM 기반 풍력발전 예측 시스템」

참고 포인트:

- 기상정보 + 시계열
- CNN 특징 추출
- LSTM 시간패턴 학습
- Hybrid prediction framework

### 김은지·이택기·김규호 (2021)
「기상 데이터를 이용한 딥러닝 기반 풍력 발전량 예측에 관한 연구」

참고 포인트:

- 기상변수와 풍력발전량 관계
- Pearson correlation
- LSTM 기반 단기 발전량 예측
- 본 연구에서는 lag correlation 및 feature ablation으로 확장

---

## 3.2 해외 레퍼런스

### Hybrid Prophet–TCN Framework for Deterministic and Probabilistic Wind-Power Forecasting

핵심 구조:

```text
과거 풍력발전량
+ 다변량 기상정보
↓
시계열 모델
↓
미래 풍력발전량
```

해당 논문의 예측 target은 일 단위 풍력발전량(MWh)이며,
본 연구는 시간 단위 발전량을 예측한다는 차이가 있습니다.

즉 연구 문제의 구조는 유사하지만 시간 해상도와 모델 구성은 다릅니다.

---

# 4. 주 데이터: 상명풍력 발전실적

## 4.1 파일

현재 확보한 파일:

```text
한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv
```

공공데이터 기반의 상명풍력 발전실적 데이터입니다.

---

## 4.2 데이터 구조

CSV는 기본적으로 wide format입니다.

예:

```text
기준일 | 호기 | 1시 | 2시 | 3시 | ... | 24시
```

시간별 발전실적을 나타냅니다.

---

# 5. 실제 CSV 검토 결과

현재 실제 파일을 직접 확인한 결과:

```text
기간:
2023-01-01 ~ 2026-03-30

총 날짜 수:
약 1,185일

시간별 유효 관측치:
약 28,368개

24시간 값이 모두 존재하는 날짜:
약 1,182일
```

일부 날짜에는 24개 시간값 전체가 결측인 경우가 존재합니다.

확인된 대표 예:

```text
2026-02-03
2026-02-15
```

이 날짜들은 그대로 0으로 처리하지 말고 결측으로 유지하거나 분석에서 제외합니다.

---

# 6. 중요: `호기` 컬럼 해석

현재 실제 CSV에서는 `호기` 값이 사실상:

```text
1
```

만 존재합니다.

따라서 이 데이터는:

> 상명풍력의 7개 개별 터빈을 구분한 데이터

라기보다

> **상명풍력 발전소 전체의 시간별 발전실적 시계열**

로 해석하는 것이 현재로서는 가장 타당합니다.

즉 본 연구의 예측 단위는:

```text
개별 터빈 발전량
```

이 아니라

```text
상명풍력발전소 전체 시간별 발전량
```

입니다.

---

# 7. 가장 중요한 데이터 이슈: 단위 스케일 변경

실제 CSV를 검토하면 2025-02-01을 전후로 값의 크기가 약 1,000배 달라집니다.

## 7.1 2023 ~ 2025-01-31

예:

```text
2,285,684
13,349,368
20,109,158
```

이 값을 Wh로 해석하면:

```text
20,109,158 Wh
= 20,109.158 kWh
= 20.109158 MWh
```

이 됩니다.

---

## 7.2 2025-02-01 이후

예:

```text
18,504.316
20,109.157
```

이 값을 kWh로 해석하면:

```text
20,109.157 kWh
= 20.109157 MWh
```

가 됩니다.

---

# 8. 물리적 검증

상명풍력의 설비용량:

```text
3 MW × 7기 = 21 MW
```

따라서 1시간 동안 생산 가능한 최대 에너지는 이론적으로:

```text
약 21 MWh
```

입니다.

단위 보정 후 실제 시간당 최대 발전량은 약:

```text
20.109 MWh
```

수준으로 설비용량과 물리적으로 일치합니다.

따라서 현재 가장 합리적인 단위 보정 규칙:

```text
2023-01-01 ~ 2025-01-31
원시값 = Wh 추정
generation_mwh = raw / 1,000,000

2025-02-01 이후
원시값 = kWh 추정
generation_mwh = raw / 1,000
```

입니다.

---

# 9. 매우 중요한 주의사항

위 단위 변경은 실제 데이터 스케일과 물리적 발전용량을 통해 강하게 추정된 결과입니다.

하지만 제공기관이 공식적으로:

```text
2025-02-01부터 Wh -> kWh로 변경
```

했다고 확인한 상태는 아닙니다.

따라서 코드에서는 다음처럼 명시적으로 처리합니다.

```python
if datetime < "2025-02-01":
    generation_mwh = raw_generation / 1_000_000
else:
    generation_mwh = raw_generation / 1_000
```

그리고 데이터 QA 문서에:

> 2025-02-01을 기준으로 약 1,000배의 scale discontinuity가 확인되며,
> 상명풍력 설비용량(21 MW)의 물리적 상한을 이용해 단위를 보정함.

이라고 기록합니다.

논문 작성 전에는 가능하면 제공기관에 단위 변경 여부를 문의합니다.

---

# 10. 단위 보정 후 연간 발전량 sanity check

단위 보정 후 대략적인 연간 발전량:

```text
2023: 약 31,331 MWh
2024: 약 30,872 MWh
2025: 약 31,655 MWh
```

연도별로 비슷한 수준이므로 단위 변환 방식이 합리적인 것으로 판단됩니다.

---

# 11. Target 정의

최종 Target:

```text
generation_mwh
```

즉:

> **상명풍력발전소의 시간별 발전량(MWh)**

입니다.

---

# 12. Wide -> Long 변환

원본:

```text
기준일 | 1시 | 2시 | ... | 24시
```

최종:

```text
datetime | generation_mwh
```

예:

```text
2023-01-01 01:00 | 2.285684
2023-01-01 02:00 | 1.240105
2023-01-01 03:00 | 0.632211
...
```

---

# 13. 24시 timestamp 처리

원본의 `24시`는 반드시 주의합니다.

예:

```text
2023-01-01 24시
```

는 실제로:

```text
2023-01-02 00:00
```

을 의미할 가능성이 높습니다.

따라서 단순히 `2023-01-01 24:00` 문자열로 만들지 말고:

```python
date + 1 day at 00:00
```

로 변환합니다.

단, 제공기관의 시간 기준 확인이 가능하면 추가 확인합니다.

---

# 14. 0값 처리

현재 시간별 발전량 중 0값이 약:

```text
17.5%
```

정도로 존재합니다.

0은 다음과 같은 여러 의미일 수 있습니다.

```text
실제 무풍
정비/고장
출력제한(curtailement)
계통 문제
측정/기록 문제
```

따라서:

```text
0 -> NaN
```

으로 자동 변환하지 않습니다.

초기에는 0을 실제 관측값으로 유지합니다.

추후 다음을 확인합니다.

- 0이 연속되는 구간 길이
- 같은 시간대 기상 풍속
- 하루 전체 0 여부
- 이상하게 긴 0 구간

이를 이용해 운영정지 가능성을 별도 flag로 분리할 수 있습니다.

---

# 15. 메인 기상관측소: 새별오름 AWS 883

전체 기간을 커버할 수 있는 메인 환경데이터로:

```text
새별오름 AWS
지점번호: 883
```

을 사용합니다.

상명풍력과 약 6~7 km 수준의 거리이며,
2023년 이전부터 운영되어 현재 발전실적 전체 기간과 정합 가능합니다.

사용할 기상변수:

```text
기온
풍속
풍향
습도
기압
강수량
```

초기 Core Model에서는:

```text
기온
풍속
풍향
습도
기압
```

을 우선 사용합니다.

---

# 16. 보조 기상관측소: 제주금악 AWS 993

상명풍력과 공간적으로 가장 가까운 관측소 후보:

```text
제주금악 AWS
지점번호: 993
```

상명풍력과 약 1~2 km 수준으로 매우 가깝습니다.

단점:

```text
관측 시작:
2024-04-06
```

따라서 2023년부터 전체 기간에는 사용할 수 없습니다.

---

# 17. 기상관측소 사용 전략

## Main Dataset

```text
상명풍력 발전량
+
새별오름 AWS 883
```

기간:

```text
2023-01-01 ~ 2026-03-30
```

---

## Secondary / Spatial Validation

공통 기간:

```text
2024-04-06 이후
```

에서:

```text
상명풍력 + 새별오름 AWS 883
```

vs

```text
상명풍력 + 제주금악 AWS 993
```

를 비교합니다.

연구 질문:

> 더 가까운 기상관측소의 기상정보가 풍력 발전량 예측 정확도를 개선하는가?

이 비교는 좋은 보조 실험이 될 수 있습니다.

---

# 18. AWS 시간 해상도

발전량 target이 1시간 단위이므로,
기상자료도 최종적으로:

```text
1시간 단위
```

로 정합합니다.

AWS가 1분 자료라면:

```text
1분 -> 1시간
```

으로 집계합니다.

---

# 19. AWS 1분 -> 1시간 집계 규칙

권장:

| 변수 | 집계 |
|---|---|
| 기온 | mean |
| 풍속 | mean |
| 습도 | mean |
| 기압 | mean |
| 풍향 | circular mean 또는 sin/cos |
| 강수량 | sum |
| 강수유무 | optional |

---

# 20. 풍향 처리

풍향은 단순 평균하지 않습니다.

예:

```text
359°
1°
2°
```

의 산술평균은 잘못된 방향을 만듭니다.

따라서:

```text
wind_dir_sin = sin(theta)
wind_dir_cos = cos(theta)
```

로 변환한 뒤 시간 평균합니다.

최종 feature:

```text
wind_dir_sin
wind_dir_cos
```

---

# 21. 최종 1차 데이터셋 구조

```text
datetime

generation_mwh        # Target

temperature
wind_speed
wind_dir_sin
wind_dir_cos
humidity
pressure
```

Optional:

```text
precipitation
rain_flag
hour_sin
hour_cos
```

---

# 22. 계절 변수

초기 연구에서는:

```text
봄 / 여름 / 가을 / 겨울
```

처럼 사람이 임의로 나눈 category를 사용하지 않습니다.

필요하면 연속적인 시간 encoding을 사용합니다.

예:

```text
hour_sin
hour_cos
day_of_year_sin
day_of_year_cos
```

단, 초기 baseline에서는 시간 encoding 없이 먼저 시작해도 됩니다.

---

# 23. 예측 문제 정의

기본:

```text
과거 발전량
+ 과거 기상정보
↓
미래 발전량
```

예:

```text
X[t-lookback : t]
↓
generation[t+h]
```

---

# 24. Forecast Horizon

1시간 단위 데이터이므로 다음 horizon을 우선 사용합니다.

```text
1시간 후   = +1
3시간 후   = +3
6시간 후   = +6
12시간 후  = +12
24시간 후  = +24
```

가능하면 논문에서는:

```text
단기/중기
```

라고만 표현하지 않고 실제 hour ahead를 숫자로 명시합니다.

---

# 25. Lookback

초기 후보:

```text
6시간
12시간
24시간
48시간
168시간(7일)
```

첫 구현 기본:

```text
lookback = 24
```

즉 과거 24시간을 이용해 미래 발전량을 예측합니다.

향후 validation을 통해 최적 window를 비교합니다.

---

# 26. 모델 입력 실험군

## M1 — Past Generation Only

```text
Past Generation
-> Future Generation
```

---

## M2 — Generation + Wind

```text
Past Generation
+ Wind Speed
+ Wind Direction
-> Future Generation
```

---

## M3 — Generation + Full Weather

```text
Past Generation
+ Wind Speed
+ Wind Direction
+ Temperature
+ Humidity
+ Pressure
-> Future Generation
```

---

## M4 — Full + Precipitation

```text
M3
+ Precipitation
```

M3 vs M4를 통해 강수 기여도를 평가합니다.

---

# 27. 추천 모델

최초 비교:

```text
Persistence
ARIMA / SARIMA
LightGBM
LSTM
CNN-LSTM
```

역할:

```text
Persistence
=> 가장 단순한 시계열 baseline

ARIMA/SARIMA
=> 전통 통계 시계열 baseline

LightGBM
=> 강력한 tabular ML baseline

LSTM
=> 주요 국내 레퍼런스와 연결

CNN-LSTM
=> 박래진 외(2022)와 연결
```

향후:

```text
TCN
Transformer
PatchTST
TimesNet
```

등으로 확장 가능합니다.

---

# 28. Feature Engineering for LightGBM

LightGBM에서는 explicit lag feature를 생성합니다.

예:

```text
generation_lag_1
generation_lag_3
generation_lag_6
generation_lag_12
generation_lag_24
generation_lag_48
generation_lag_168
```

Rolling features:

```text
generation_roll_mean_3
generation_roll_mean_6
generation_roll_mean_24
generation_roll_std_24
```

기상 변수에도 lag를 만들 수 있습니다.

---

# 29. LSTM / CNN-LSTM

LSTM에서는 explicit lag column보다:

```text
[batch, time, feature]
```

형태의 sequence를 사용합니다.

예:

```text
X shape:
(N, 24, F)

Y shape:
(N, 1)
```

24시간 lookback + 1시간 ahead 기준입니다.

---

# 30. 데이터 누수 방지

예:

```text
t = 2024-01-01 10:00
target = 11:00 발전량
```

사용 가능:

```text
발전량 <= 10:00
실제 AWS <= 10:00
```

사용 불가:

```text
11:00 실제 풍속
11:00 실제 기온
```

왜냐하면 10:00 시점에서는 미래 실제 관측값을 알 수 없기 때문입니다.

---

# 31. 매우 중요한 Forecasting 설계 구분

## Retrospective / Ex-post Prediction

미래 실제 기상값을 feature로 넣으면:

```text
Actual Weather_t+h
```

이므로 이는 실제 운영 forecasting과 다릅니다.

이 실험을 한다면 반드시:

> retrospective prediction / ex-post analysis

로 표현합니다.

---

## Realistic Forecasting

실제 운영 예측에서는:

```text
Past Weather
+
Future Weather Forecast
```

가 필요합니다.

향후:

```text
LDAPS / GFS / NWP
```

기상예보 데이터를 추가하면 현실적인 발전량 forecasting으로 확장 가능합니다.

---

# 32. Train / Validation / Test

랜덤 split을 사용하지 않습니다.

초기 추천:

```text
Train:
2023-01-01 ~ 2024-06-30

Validation:
2024-07-01 ~ 2024-12-31

Test:
2025-01-01 ~ 2025-12-31
```

2026 데이터는:

```text
Final Holdout / additional external temporal test
```

로 남기는 방식을 권장합니다.

단, 2025-02-01 단위 변경 이슈가 있으므로 단위 정규화가 먼저 완료되어야 합니다.

---

# 33. 추천되는 더 보수적인 split

단위 변경 이슈를 고려하면 다음도 고려:

```text
Train:
2023-01-01 ~ 2024-06-30

Validation:
2024-07-01 ~ 2024-12-31

Test A:
2025-01-01 ~ 2025-01-31

Test B:
2025-02-01 ~ 2025-12-31
```

목적:

> 단위 변경 전후 전처리가 모델 성능에 영향을 주지 않는지 확인

최종적으로 단위 보정 후 distribution continuity가 충분하면 2025 전체를 하나의 Test로 합쳐도 됩니다.

---

# 34. 평가 지표

기본:

```text
MAE
RMSE
R²
```

추가:

```text
nMAE
MAPE / sMAPE
```

주의:

발전량이 0인 구간이 많기 때문에 MAPE는 문제가 될 수 있습니다.

따라서:

```text
MAE
RMSE
nMAE
sMAPE
```

를 우선 권장합니다.

---

# 35. 0 발전량 처리

현재 약 17.5%가 0입니다.

반드시 다음 EDA 수행:

```text
0 연속 길이
0이 발생하는 시간대
0이 발생할 때 풍속
일 전체가 0인 날짜
주/월별 0 비율
```

강풍인데 발전량이 0이면:

```text
정비
고장
curtailment
계통 제한
```

가능성이 있으므로 별도 anomaly 후보로 봅니다.

---

# 36. 상관관계 분석

동시간:

```text
wind_speed vs generation
temperature vs generation
humidity vs generation
pressure vs generation
```

추천:

```text
Pearson
Spearman
```

---

# 37. Lagged Correlation

예:

```text
wind_speed_t
vs
generation_t+1

wind_speed_t
vs
generation_t+3

wind_speed_t
vs
generation_t+6

wind_speed_t
vs
generation_t+24
```

이를 통해 기상정보의 시간적 선행성을 분석합니다.

---

# 38. Granger Causality

사용 가능하지만 해석에 주의합니다.

표현:

```text
predictive causality
시간적 선행성
```

금지:

```text
물리적 인과관계가 증명되었다
```

---

# 39. 새별오름 vs 제주금악 비교

2024-04-06 이후 공통 기간에서:

```text
Model_A
= Generation + Saebyeol Weather

Model_B
= Generation + Jeju-Geumak Weather
```

를 비교합니다.

연구 질문:

> 발전소와 더 가까운 기상관측소의 자료가 발전량 예측 성능을 개선하는가?

이 실험은 보조 연구 결과로 상당히 유용합니다.

---

# 40. 데이터 병합 전 QA

## Generation

- [ ] CSV encoding 확인
- [ ] 날짜 중복 확인
- [ ] 호기 unique 확인
- [ ] 24개 시간컬럼 결측률
- [ ] 날짜별 24시간 완전성
- [ ] raw scale change 자동 탐지
- [ ] 단위 변환 후 최대값 <= 21 MWh 확인
- [ ] 0값 비율
- [ ] 연속 0 길이
- [ ] 이상치

## Weather

- [ ] 883 기간 확인
- [ ] 993 기간 확인
- [ ] timestamp 중복
- [ ] 결측률
- [ ] 시간 gap
- [ ] 각 변수 물리적 이상치
- [ ] 관측소 위치 변경 여부

## Merge

- [ ] datetime matching rate
- [ ] merge 전/후 행 수
- [ ] feature별 missing rate
- [ ] longest gap

---

# 41. 결측치 처리 원칙

### 발전량 Target

보간하지 않습니다.

```text
generation_mwh = NaN
```

이면 해당 target 샘플을 제거합니다.

---

### 기상정보

짧은 결측은 interpolation 후보입니다.

단:

```text
허용 gap threshold
```

를 EDA 후 결정합니다.

긴 결측은 해당 시점/window를 제거합니다.

---

# 42. 권장 최종 테이블

```text
datetime
generation_mwh

temperature
wind_speed
wind_dir_sin
wind_dir_cos
humidity
pressure

precipitation       # optional
weather_station     # optional
```

---

# 43. 권장 디렉토리

```text
wind-generation-forecast/
│
├─ README.md
├─ requirements.txt
│
├─ configs/
│  └─ base.yaml
│
├─ data/
│  ├─ raw/
│  │  ├─ generation/
│  │  │  └─ sangmyeong_wind_generation.csv
│  │  │
│  │  └─ weather/
│  │     ├─ saebyeol_883/
│  │     └─ jeju_geumak_993/
│  │
│  ├─ interim/
│  │  ├─ generation_hourly.parquet
│  │  ├─ saebyeol_hourly.parquet
│  │  ├─ geumak_hourly.parquet
│  │  └─ merged.parquet
│  │
│  └─ processed/
│     ├─ train.parquet
│     ├─ valid.parquet
│     └─ test.parquet
│
├─ notebooks/
│  ├─ 01_generation_eda.ipynb
│  ├─ 02_weather_eda.ipynb
│  ├─ 03_unit_normalization.ipynb
│  ├─ 04_merge.ipynb
│  ├─ 05_correlation_lag.ipynb
│  └─ 06_baseline_models.ipynb
│
├─ src/
│  ├─ data/
│  │  ├─ load_generation.py
│  │  ├─ clean_generation.py
│  │  ├─ load_weather.py
│  │  ├─ aggregate_weather.py
│  │  └─ merge.py
│  │
│  ├─ features/
│  │  ├─ wind_direction.py
│  │  ├─ lag_features.py
│  │  └─ time_features.py
│  │
│  ├─ models/
│  │  ├─ persistence.py
│  │  ├─ arima.py
│  │  ├─ lightgbm_model.py
│  │  ├─ lstm.py
│  │  └─ cnn_lstm.py
│  │
│  └─ evaluation/
│     ├─ metrics.py
│     └─ plots.py
│
└─ reports/
   ├─ figures/
   └─ tables/
```

중간 파일은 CSV보다 parquet 권장.

---

# 44. Codex 첫 작업 순서

## Step 1

현재 상명풍력 CSV 재검증

출력:

```text
shape
columns
date range
unique turbine values
missing values
zero ratio
raw min/max
daily completeness
```

---

## Step 2

Wide -> hourly long 변환

```text
date + hour
->
datetime
```

24시 처리 주의.

---

## Step 3

단위 normalization

```text
before 2025-02-01:
Wh -> MWh

after 2025-02-01:
kWh -> MWh
```

---

## Step 4

Sanity checks

```text
generation_mwh < 0
generation_mwh > 21
yearly generation
monthly generation
```

---

## Step 5

0값 EDA

```text
zero streak
zero by month
zero by hour
```

---

## Step 6

새별오름 AWS 883 로딩/EDA

필요 컬럼:

```text
datetime
temperature
wind_speed
wind_direction
humidity
pressure
precipitation
```

---

## Step 7

1시간 weather aggregation

풍향 sin/cos 처리.

---

## Step 8

Generation + Weather merge

```text
datetime
```

기준 inner/left join 결과 비교.

---

## Step 9

QA Report 생성

```text
match rate
missing %
date range
gap
```

---

## Step 10

Baseline

먼저:

```text
Persistence
ARIMA
LightGBM
```

이후:

```text
LSTM
CNN-LSTM
```

---

# 45. Codex용 시작 프롬프트

아래 내용을 그대로 사용할 수 있습니다.

```text
이 프로젝트는 상명풍력발전소의 시간별 풍력 발전량을 예측하는 시계열 연구이다.

README의 현재 연구계획을 기준으로 작업하라.

주 Target:
generation_mwh

주 발전 데이터:
한국중부발전 상명풍력 시간별 발전실적

주 기상 데이터:
새별오름 AWS 883

보조 기상 데이터:
제주금악 AWS 993 (2024-04-06 이후)

가장 먼저 다음을 수행하라.

1. 상명풍력 CSV의 shape, 기간, 컬럼, 호기 unique, 결측, 0값, 이상치 재검증
2. 1시~24시 wide format을 시간별 long format으로 변환
3. 24시 timestamp를 다음날 00:00으로 올바르게 변환
4. 2025-02-01 전후 약 1,000배 scale discontinuity를 검증
5. 2025-02-01 이전 raw 값은 Wh, 이후는 kWh로 가정하여 MWh로 통일
6. 변환 후 시간당 발전량이 21 MWh를 초과하는지 검사
7. 연간/월간 발전량 sanity check
8. 0 발전량의 연속 구간과 비율 분석
9. 새별오름 AWS를 1시간 단위로 집계
10. 발전량과 기상정보를 datetime 기준으로 병합
11. 병합률과 결측률 QA report 생성

주의:
- 0 발전량을 자동으로 결측 처리하지 말 것
- 미래 실제 AWS 값을 실시간 forecasting 입력으로 사용하는 설계는 data leakage가 될 수 있음을 명시할 것
- 풍향은 단순 평균하지 말고 sin/cos 방식으로 처리할 것
- 발전량 단위 보정은 반드시 원 raw 값과 변환값을 함께 보존할 것
- 단위 변경은 제공기관 공식 확인 전까지 추정임을 metadata에 기록할 것
- 랜덤 train/test split을 사용하지 말 것
```

---

# 46. 현재 확정 사항

```text
Research target:
풍력 발전량 예측

Main generation data:
상명풍력

Time resolution:
1시간

Target:
generation_mwh

Main weather:
새별오름 AWS 883

Secondary weather:
제주금악 AWS 993

Initial lookback:
24시간

Forecast horizons:
1h / 3h / 6h / 12h / 24h

Initial models:
Persistence
ARIMA/SARIMA
LightGBM
LSTM
CNN-LSTM
```

---

# 47. 아직 확인이 필요한 항목

- [ ] 제공기관의 실제 상명풍력 단위 정의
- [ ] 2025-02-01 단위 변경 여부 공식 확인
- [ ] `호기=1`이 발전소 전체를 의미하는지
- [ ] `24시`의 공식 timestamp 의미
- [ ] 0 발전량의 운영상 의미
- [ ] 상명풍력과 AWS의 정확한 거리
- [ ] 새별오름 AWS의 전체 기간 품질
- [ ] 제주금악 AWS의 2024-04-06 이후 결측률
- [ ] 실제 forecasting에 NWP(LDAPS/GFS)를 추가할지 여부

---

# 48. 연구 제목 후보

초기 작업명:

> **기상정보와 시계열 학습을 활용한 풍력발전소 시간별 발전량 예측**

조금 더 구체적으로:

> **인접 기상관측 정보와 시계열 모델을 활용한 상명풍력발전소 시간별 발전량 예측**

모델 확정 후 제목은 다시 조정합니다.

---

# 49. 가장 중요한 원칙

이 연구의 중심은:

> **얼마나 복잡한 모델을 쓰느냐**

보다:

> **발전량과 기상정보를 물리적/시간적으로 올바르게 정합하고, 미래정보 누수 없이 공정하게 모델을 비교하는 것**

입니다.

데이터 단위, timestamp, 결측, 0값, forecast horizon을 먼저 확정한 뒤 모델을 복잡하게 만듭니다.
