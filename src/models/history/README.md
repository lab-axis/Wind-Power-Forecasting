# EMFN (Exogenous Multiscale Fusion Network) Architecture Evolution History

이 디렉토리는 상명풍력 시간별 발전량 예측 연구에서 제안 모델 **EMFN**이 거쳐온 3단계 구조 진화 과정을 보존하는 아카이브입니다.
최종 확정된 메인 모델 코드는 상위 디렉토리의 [`src/models/emfn.py`](../emfn.py) 및 [`src/models/emfn_trainer.py`](../emfn_trainer.py)에 위치합니다.

---

## 1. 버전별 구조 진화 개요

| 버전 (Version) | 파일명 | 핵심 아키텍처 및 특징 | 주요 한계 및 해결 과제 |
|:---|:---|:---|:---|
| **EMFN v1** | `emfn_v1.py`<br>`emfn_v1_trainer.py` | - 단일 통합 Dilated Causal TCN 백본<br>- 발전량과 기상 6종을 단순 채널 결합(Concat)<br>- Bernoulli-Beta Hurdle 출력 헤드 적용 | **음의 전이(Negative Transfer)** 발생:<br>기상 정보 결합 시 단변량 모델 대비 초단기(+1h) MAE 악화 (기상 표현이 발전량 고유의 시계열 자기상관을 희석시킴) |
| **EMFN v2** | `emfn_v2.py`<br>`emfn_v2_trainer.py` | - Dual Encoders (발전량 인코더 + 기상 인코더 분리)<br>- Cross-Attention 메커니즘을 통한 기상 특징 주입<br>- 독립된 기상-발전량 상호작용 설계 | 음의 전이는 성공적으로 해소(+3.7% 성능 개선)되었으나, 복잡한 Cross-Attention 연산 비용 대비 +1h 초단기에서 GBDT(LightGBM) 대비 여전히 오차 격차가 잔존함 |
| **EMFN v3**<br>*(메인 모델로 승격)* | `emfn_v3.py`<br>`emfn_v3_trainer.py`<br>$\rightarrow$ `../emfn.py` | - **Multi-channel Causal TCN Backbone**<br>- **Task-Decoupled Dual-Route Network** (발전량 볼륨 분기 vs 영발전 분류 분기 완전 분리)<br>- **High-Frequency Residual Skips**: $y_t$, 모멘텀 $(y_t - y_{t-1})$, 단기 변동성 직결 스킵<br>- **Selective Weather Gating**: Cut-in 풍속 인접도 및 연속 0발전 스트릭 동역학 반영<br>- **Composite Supervised Hurdle Objective**: NLL + Huber 정밀 보정 | **최종 제안 모델 (Proposed Model)**:<br>- 음의 전이 완전 극복<br>- 1h ~ 12h 일중 급전 스펙트럼 전 구간 딥러닝 1위 석권<br>- 물리적 유계 위반율 **0.00% 자체 보장** |

---

## 2. 파일 보존 목적
- 학술 논문 제5장 및 제6장의 **모델 구조 진화 분석(Architecture Evolution Analysis)** 및 **어블레이션 실험(Ablation Study)** 결과의 수치적 재현성을 영구히 보장하기 위함입니다.
