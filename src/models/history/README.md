# EMFN 구조 이력

현재 모델은 `../emfn.py`와 `../emfn_trainer.py`의 v3입니다.

2026-10-05 기준, v3의 구조는 유지한 채 학습 선택 기준을 검증 CRPS로 통일한 3개 시드 실험을 완료했습니다. 과거 재현용 벤치마크/어블레이션 진입점은 loss 선택을 명시합니다. 이는 새 구조 버전 추가가 아니라 학습·평가 규약의 수정입니다. [실험 이력](../../../reports/EXPERIMENT_HISTORY.md)과 [현재 논문 구성안](../../../reports/baseline_benchmarks_and_future_plan.md)을 구분하여 참고하십시오.

| 버전 | 실제 소스 구조 | 해석 |
|---|---|---|
| v1 | 시계열 분해, 발전량·기상 Dual Encoders, Cross-Attention Fusion | 분리 인코더와 교차 주의 기반 결합 |
| v2 | Multi-channel Causal TCN, Weather State Gating | 채널 결합 기반 백본 |
| v3 | 공유 Multi-channel TCN, magnitude/zero adapters, HF/AR 경로, 선택적 기상 특징 MLP, Hurdle-Beta | 현재 유지하는 메인 구조 |

이전 README는 v1/v2 구조를 서로 바꾸어 기술했습니다. 위 표는 소스에 맞게 정정했습니다.
v3의 두 분기는 공유 백본을 사용하므로 완전히 독립적인 네트워크가 아닙니다.
`z_zero`라는 변수명과 달리 코드에서 `sigmoid(z_zero)`는 **양수 발전 확률**입니다.
TCN의 causal은 입력의 시간 방향 처리이며 기상 변수의 인과 효과를 식별한다는 뜻이 아닙니다.
게이팅은 통계적 특징 변환이며 제조사 시동 풍속이나 터빈 제어 법칙이 내장되어 있지 않습니다.

역사 파일은 변경하지 않았습니다. 다만 현재의 공용 전처리·평가 모듈과 함께 실행하면
과거 실행 환경을 그대로 재현하는 것이 아닙니다. 과거 학습의 정확한 코드/시드/환경이
기록되지 않은 경우 확인 불가입니다. 수정 전 전체 스냅샷과 파일 해시는
`reports/archive/pre_correction_manifest.json` 및 로컬 `.experiment_archive/`에서 확인합니다.
기존 수치는 구 평가 파이프라인의 결과이며 현재 우월성의 근거로 사용하지 않습니다.
