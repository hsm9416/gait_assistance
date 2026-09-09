# gait_assistance

각 보행 주기(stride)를 **Log-Euclidean SPD 다양체(manifold)** 위의 점으로 표현하여
편마비 보행을 실시간으로 보조하는 패키지입니다.

지도 학습 분류기나 사전에 녹화된 환자 데이터는 어디에도 사용하지 않습니다.
환자 모델은 세션 초반의 스트라이드로부터 **환자별 베이스라인(patient-specific
baseline)** 을 만들어 온라인으로 구축되며, 유일한 오프라인 산출물은 선택 사항인
정상인 기준(*healthy reference*)뿐입니다.

> **용어 주의.** 세션 초반 스트라이드는 "그 환자의 정상"이 **아닙니다**.
> 베이스라인은 그 환자가 그 세션에서 실제로 어떻게 걸었는지를 기술한 것일 뿐,
> 정상성의 기준이 아닙니다. 편마비 환자의 베이스라인은 정의상 병적 패턴이며,
> 베이스라인 위에 정확히 얹힌 스트라이드도 여전히 심하게 병적일 수 있습니다.
> 그래서 이 패키지는 **베이스라인 편차**와 **병적 편차**를 끝까지 분리해서
> 계산하고 기록합니다.

## 동작 원리

각 스트라이드는 SPD 다양체 위의 한 점으로 변환된 뒤, 세 개의 기준점과 비교됩니다.

```
sensors -> estimated phase -> stride -> resample(0-100%) -> z-score(baseline)
        -> C = Cov(X) + eps*I -> Z = log(C) = V diag(log lambda) V^T
        -> nearest cluster centroid -> gait state + confidence + OOD
        |
        +-- E_R  = clip((||Z - Z_H||_F - healthy_threshold) / scale, 0, 1)
        |          severity/context only, 0 inside the healthy region
        |
        +-- E_B  = w_s * swing_ratio_deficit + w_b * belt_excursion_deficit
        |          device-actionable deficit, 0 inside the healthy intervals
        |
        -> E_assist = E_B * (base_factor + manifold_factor * E_R)
        -> decision state -> persistence gate -> rate limit
        -> assist_gain -> target_belt_length -> impedance -> current
```

**보조력은 swing 구간에서만 들어갑니다.** stance에서는 프로파일 항이 0이라
`target = baseline_belt_length`가 되어 장치가 중립 길이를 유지합니다. 상위
루프가 어떤 게인을 게시했든 마찬가지이므로, 게인은 "다음 swing을 얼마나 도울지"에
대한 값이지 상시 당김이 아니며 입각기의 발에 하중을 걸지 않습니다. 이 규칙은
`BeltTargetGenerator.assists_in()`에 있으며 호출자가 아니라 이 모듈이 강제합니다.
(`baseline_belt_length`는 환자 모델에서 나오므로 모델이 만들어지기 전에는 목표가
측정 위치를 따라갑니다 — "모델이 생기기 전에는 현재 위치를 유지한다" 절 참고.)

**보조량은 `E_B`가 결정하고 `E_R`은 그 강도를 조절할 뿐입니다.** `E_B = 0`이면
`E_assist = 0`이므로, 다변량 패턴이 아무리 정상 분포에서 벗어나 있어도 장치가
교정할 수 있는 결손이 없으면 보조하지 않습니다. Riemannian 거리를 모터 토크에
직접 대응시키지 않는다는 것이 이 구조의 요점입니다.

### Healthy reference는 점이 아니라 영역이다

정상인 데이터의 각 stride에 대해 `Z_i = log(C_i)`, `Z_H = mean(Z_i)`,
`d_i = ||Z_i - Z_H||_F`를 계산하고, 그 분포로 **영역 경계**를 만듭니다.

```
healthy_distance_threshold = percentile(d_i, 95)   # 기본값
```

현재 stride가 `d_H = ||Z_t - Z_H||_F <= threshold` 이면 `E_R = 0`입니다.
바깥으로 나간 초과분만 평가합니다.

```
E_R = clip((d_H - healthy_distance_threshold) / manifold_scale, 0, 1)
```

정상인 보행도 하나의 점이 아니라 분포이므로, **환자를 healthy centroid까지
끌고 가지 않습니다.** 정상 범위 안에 들어오면 그것으로 충분합니다.

임계값 계산법은 `reference.threshold_method`로 바꿀 수 있습니다.

| 값 | 식 | 비고 |
|---|---|---|
| `percentile` (기본) | `percentile(d_i, 95)` | 분포 가정 없음, 이상치에 강함 |
| `mean_std` | `mean + sigma*std` | 이전 버전의 방식 |
| `median_mad` | `median + k*MAD` | 가장 robust |

### 세 가지 target mode

`deviation.target_mode`로 선택하며 기본값은 `healthy_region`입니다.

| 모드 | `Z_target` | 0-error 영역 |
|---|---|---|
| `healthy_region` (기본) | `Z_H` | 있음 — region 내부는 `E_R = 0` |
| `interpolated_target` | `(1-alpha)*Z_patient + alpha*Z_healthy` | 없음 (호환 모드) |
| `patient_baseline` | `Z_patient` | 없음 |

healthy reference가 없으면 요청한 모드와 무관하게 `patient_baseline`으로
내려가고, 그 사실이 `warnings`에 남습니다.

### 두 종류의 편차를 분리한다

거리 세 개는 서로 다른 질문에 답하므로 절대 하나로 합치지 않습니다.

| 값 | 기준점 | 의미 | 없을 때 |
|---|---|---|---|
| `d_patient` | `Z_patient` | **베이스라인 편차**. 그 환자 자신의 초반 스트라이드로부터 얼마나 벗어났는가 = 세션 내 일관성. 중증도에 대해서는 아무 말도 하지 않음 | 항상 정의됨 |
| `d_healthy` | `Z_H` | **병적 편차(pathological deviation)**. 정상인 기준으로부터 얼마나 벗어났는가. 병리를 말할 수 있는 유일한 값 | healthy reference 없으면 **NaN** (절대 `d_patient`로 대체하지 않음) |
| `manifold_deviation` (`E_R`) | healthy **region** | 위 `d_healthy`를 region 경계 기준으로 정규화한 severity/context 점수 | healthy reference 없으면 `d_target` 기반 fallback |

`d_patient`는 **보조 트리거가 아닙니다.** 세션 중 변화·피로·개선량을 보는 추세
지표이며, 보조량은 `E_B`(생체역학 결손)가 결정합니다.

`stride_table.csv`에는 이전 버전의 가중합 항(`e_manifold`, `e_baseline`,
`e_healthy`, `deviation_score`)도 남아 있지만 **비교용 기록일 뿐 게인에
관여하지 않습니다.** 실제 게인은 `raw_assist_gain` / `assist_gain` 컬럼이고
그 근거는 `biomechanical_deviation`과 `manifold_deviation`입니다.

## 두 가지 보조 모드

모드는 사용자가 고르는 값이 아니라 **healthy reference의 유무로 결정**됩니다.
두 모드는 편차 점수가 뒷받침할 수 있는 주장 자체가 다르기 때문입니다.

| | `BASELINE_STABILIZATION` | `HEALTHY_DIRECTED` |
|---|---|---|
| 조건 | healthy reference 없음 | healthy reference 있음 |
| target mode | `patient_baseline`으로 강제 | `healthy_region` (기본) |
| 보조의 목표 | 환자를 **자기 베이스라인 주위로 안정화** | 환자를 **정상 범위 안으로 유도** |
| `d_healthy` / `E_R` | **NaN / fallback** | region 기준 병적 편차 |
| 생체역학 정상 구간 | 환자 본인 베이스라인의 5~95 백분위 | 정상인 데이터셋의 5~95 백분위 |
| `reference_source` | `patient_baseline` | `healthy_reference` |

`BASELINE_STABILIZATION`에서 나온 편차 점수를 중증도나 회복 지표로 읽으면
안 됩니다. 그 모드에서 점수가 낮다는 것은 "오늘 일관되게 걸었다"는 뜻이지
"정상에 가깝다"는 뜻이 아닙니다. `reference_source` 컬럼이 매 stride 어느
기준으로 판정했는지 남기므로 둘을 혼동할 여지가 없습니다.

모드는 `patient_model.json`, 요약 JSON, `stride_table.csv`의 `assist_mode` 컬럼,
그리고 실행 로그의 `[mode]` 줄에 모두 찍힙니다.

## 생체역학 지표: 평가용과 보조 결정용을 분리한다

모든 지표를 보조량 계산에 넣지 않습니다. 두 그룹으로 나뉩니다.

**Evaluation metrics** (연구·로깅 전용, 게인에 들어가지 않음)

`estimated_stance_ratio`, `estimated_swing_stance_ratio`, temporal symmetry,
trunk motion, `d_patient`, `d_healthy`, cluster/confidence.

**Device-actionable metrics** (실제 보조를 결정)

기본값은 swing ratio deficit과 belt excursion deficit입니다. 가중치는
`deviation.biomech_weights`로 조정합니다.

```python
biomech_weights = {
    "swing_ratio": 0.5,
    "belt_excursion": 0.5,
    "temporal_symmetry": 0.0,   # 좌우 데이터가 있을 때만 활성화
    "trunk_compensation": 0.0,
}
```

가중치를 0으로 두면 그 지표는 **계산되고 기록되지만 게인에 반영되지 않습니다.**
연구 지표가 조용히 제어 입력이 되는 것을 막는 장치입니다.

### 정상 평균이 아니라 정상 범위 밖만 평가한다

각 지표의 정상 범위는 정상인 분포의 5~95 백분위입니다
(`reference.metric_lower_percentile` / `metric_upper_percentile`).

```
L <= x <= U        -> error = 0
x < L              -> error = (L - x) / scale
x > U              -> error = (x - U) / scale
```

단 **장치의 작용 방향과 연결된 지표는 방향성 error**를 씁니다. 이 장치는 swing
excursion 부족을 보완하므로 `swing_ratio`와 `belt_excursion`은 `deficit` 방향만
셉니다. 정상보다 excursion이 *큰* 경우에는 보조량을 올리지 않습니다.

| term | 읽는 지표 | 방향 |
|---|---|---|
| `swing_ratio` | `swing_ratio` | deficit (하한 미만만) |
| `belt_excursion` | `belt_excursion` | deficit (하한 미만만) |
| `temporal_symmetry` | `swing_symmetry_ratio` | two-sided |
| `trunk_compensation` | `trunk_acc_rms` | excess (상한 초과만) |

측정되지 않은 지표나 정상 범위가 없는 지표의 error는 **0이 아니라 `None`** 이며,
`E_B`에 기여하지 않습니다. 결과적으로 보조량이 줄어드는 쪽 — 안전한 방향입니다.

## 좌우 대칭성

paretic / non-paretic 양쪽 gait event가 모두 있을 때만 계산합니다.

```
swing_symmetry_ratio     = swing_time_paretic / swing_time_nonparetic
stance_symmetry_ratio    = stance_time_paretic / stance_time_nonparetic
ss_ratio_paretic         = swing_time_paretic / stance_time_paretic
ss_ratio_nonparetic      = swing_time_nonparetic / stance_time_nonparetic
ss_symmetry_ratio        = ss_ratio_paretic / ss_ratio_nonparetic
```

이 장치는 한쪽 다리만 계측하므로 기본 실행에서는 전부 `None`이고 로그에 빈 칸으로
남습니다. **반대측 값을 인구 평균이나 미러링으로 채우지 않습니다** — 추정으로
만든 대칭성 비율은 측정값과 구별되지 않은 채 보조량을 움직이게 되기 때문입니다.

반대측을 계측하려면 `gait.symmetry.ContralateralTimingSource`를 구현해
`compute_stride_metrics(stride, contralateral=...)`로 넘기면 됩니다.

## 보조 결정 상태

매 stride마다 다음 중 하나로 분류되어 `decision_state` 컬럼에 기록됩니다.                                                                                 

| 상태 | 조건 | 보조 |
|---|---|---|
| `IN_RANGE` | `E_R = 0`, `E_B = 0` | 없음 |
| `BIOMECH_DEFICIT_ONLY` | `E_R = 0`, `E_B > 0` | 기본 수준 (`base_factor`) |
| `MANIFOLD_DEVIATION_ONLY` | `E_R > 0`, `E_B = 0` | **없음** — 로그·감시만 |
| `COMBINED_DEVIATION` | `E_R > 0`, `E_B > 0` | 강화 |
| `OOD` | 기존 OOD 조건 | 강한 보조 금지 (hold 또는 safe minimum) |

OOD에서의 동작은 `assist.ood_policy`로 고릅니다: `hold`(기본)는 직전 게인을
유지하고 `safe_minimum`은 `assist.ood_safe_gain`으로 내려갑니다. 어느 쪽이든
`assist.ood_max_gain`을 넘지 않습니다.

## 연속 stride 조건 (persistence)

한 stride의 노이즈로 보조가 오르내리지 않도록, 게인이 *움직일 수 있는지* 자체를
따로 판정합니다.

```python
required_consecutive_deficit_strides = 3    # 이만큼 연속 결손이어야 게인 증가
required_consecutive_recovery_strides = 3   # 이만큼 연속 정상이어야 게인 감소
```

rate limit(`max_gain_delta`)과 함께 작동하지만 역할이 다릅니다. rate limit은
게인이 *얼마나 빨리* 움직일 수 있는지를, persistence gate는 *움직여도 되는지*를
결정합니다.

## 세션 중 변화 추적

`d_patient`는 보조 트리거의 주 기준이 아니라 **추세 지표**입니다. 세션 중 환자
변화, 피로/악화, 베이스라인 대비 개선량을 보는 용도이며 로그에 함께 남습니다.

```
delta_healthy = baseline_healthy_distance - current_healthy_distance
```

양수면 세션 시작 시점보다 healthy region에 가까워진 것입니다.

## 보행 속도 조건화 (인터페이스만 준비)

swing/stance ratio는 보행 속도의 영향을 받으므로 healthy reference가 향후
speed-conditioned reference를 지원할 수 있도록 인터페이스를 열어 두었습니다.

```python
HealthyReference.get_metric_range(metric="swing_ratio", walking_speed=current_speed)
```

현재 구현은 속도를 무시하고 전체 population 구간을 반환합니다
(`speed_conditioned = False`). 향후 speed bin / nearest speed group /
regression-based expected value 중 어느 방식으로 확장하더라도 호출부는 바뀌지
않습니다.

행렬 로그는 `scipy.linalg.logm`이 아니라 항상 `numpy.linalg.eigh`로 계산하고,
고유값을 `epsilon`에서 클리핑하기 때문에 특이행렬에 가까운 공분산이 들어와도
`-inf`가 생기지 않습니다. 대칭 행렬은 비대각 성분에 `sqrt(2)`를 곱해 벡터화하며,
이렇게 하면 벡터 공간의 유클리드 거리가 행렬의 프로베니우스 거리와 같아집니다.
덕분에 벡터 공간에서의 k-means가 그 자체로 Log-Euclidean k-means가 됩니다.

### 두 개의 루프

| | 주기 | 역할 |
|---|---|---|
| `LowLevelLoop` | 100-1000 Hz | 데이터 획득, 안전 로직, 보행 위상 **추정**, 벨트 목표값 보간, 임피던스 -> 전류 |
| `HighLevelLoop` | 0.5-2 Hz (스트라이드마다) | 특징 추출, 공분산, 로그, 보행 상태, 편차, 보조 게인 |

느린 루프에서 빠른 루프로 넘어가는 데이터는
`(assist_gain, baseline_belt_length)` 뿐이며, `AssistCommand`가 락을 걸고 게시합니다.
상위 루프는 별도 스레드에서 실행할 수 있고, 큐가 가득 차면 제어 루프를 막는 대신
해당 스트라이드를 버립니다.

**리만 기하 계층은 토크나 전류를 직접 만들지 않습니다.** 게인과 벨트 설정값만
게시하며, 이를 전류로 바꾸는 것은 오직 임피던스 제어기입니다.

### 모델이 생기기 전에는 현재 위치를 유지한다

`baseline_belt_length`는 환자 모델에서 나오므로 베이스라인 수집이 끝나기
전까지는 존재하지 않습니다. `AssistCommand`의 초기값은 `0.0 mm`이지만, 착용
상태의 벨트는 장치에 따라 `-100 mm` 부근에 있습니다. 그 기본값을 그대로 목표로
쓰면 임피던스 제어기가 벨트를 0 mm로 끌어당기며, 보조 게인이 0인 동안에도
베이스라인 수집 내내 한 방향 전류가 나갑니다.

그래서 `LowLevelLoop.step_with_sample()`은 모델이 아직 없을 때
(`AssistCommand`의 `stride_id < 0`) 목표를 **측정된 벨트 길이 자체로** 둡니다.

```python
gain, baseline_belt, stride_id = self.assist.read()
if stride_id < 0:
    baseline_belt = sample.belt_length     # 위치 오차 0 -> 위치 항 전류 0
```

위치 오차가 0이 되므로 스프링 항은 전류를 만들지 않습니다. damping 항은 그대로
남아 벨트 속도에 반응하며, 이는 임피던스 제어기의 정상 동작입니다.

모델이 완성되는 순간 `AssistCommand.publish()`가 실제
`model.baseline.baseline_belt_length`를 게시하고, 목표는 그 값으로 넘어갑니다.
게인은 이때도 `0.0`으로 게시되므로 보조는 그다음 분석 stride부터 시작합니다.

회귀 테스트는 `tests/test_controller.py`의
`test_low_level_loop_holds_position_until_the_model_exists`입니다.

## 구성

```
gait_assistance/
  main.py              CLI: offline | healthy | inspect | simulate | live
  config.py            모든 튜닝 파라미터, 중첩 dataclass + JSON
  state_machine.py     INIT..SAFE_STOP, 가드 조건이 있는 상태 전이
  loops.py             HighLevelLoop, LowLevelLoop, TwoLoopRuntime
  offline_sim.py       CSV 리플레이 파이프라인
  plotting.py          결과 그림 (PCA는 표시 전용)
  sensors/             imu.py, encoder.py (+ MockMotor/PadMotor), sensor_manager.py
  gait/                phase_detector.py (scheduled placeholder + seam),
                       stride_segmenter.py,
                       feature_extractor.py (GaitMetrics), normalization.py,
                       symmetry.py (좌우 대칭, 반대측 있을 때만)
  manifold/            covariance.py, log_euclidean.py, clustering.py,
                       reference.py (healthy region + MetricRange)
  patient/             baseline.py, patient_model.py
  control/             assistance_policy.py (E_R/E_B, decision state,
                       persistence), target_generator.py,
                       impedance_controller.py, safety.py
  utils/               logger.py, filters.py, timing.py
  tests/               test_covariance, test_log_euclidean, test_clustering,
                       test_phase, test_controller, test_assistance,
                       test_scheduled_phase
```

## 사용법

패키지가 들어 있는 디렉터리(저장소 루트)에서 실행합니다.

```bash
# 1. 녹화 데이터 리플레이: 베이스라인 -> 클러스터링 -> 온라인 리플레이 -> 게인 -> 그래프
python -m gait_assistance.main offline csv/imu_hs_torque_20260826_110500.csv \
    --out-dir results

# 2. 정상인 녹화 데이터로 만든 healthy reference와 함께 실행
python -m gait_assistance.main offline csv/patient.csv \
    --healthy-csv csv/healthy.csv --out-dir results

# 3. 재사용 가능한 healthy reference 저장
python -m gait_assistance.main healthy csv/healthy.csv --out healthy_reference.npz

# 4. 녹화 데이터의 스트라이드 분할 확인 (위상 검출기 튜닝)
python -m gait_assistance.main inspect csv/patient.csv

# 5. 하드웨어 없이: mock 센서 + mock 모터, OOD 유발용 드리프트 포함
python -m gait_assistance.main simulate --duration 120 --fast --drift-after 60

# 6. 실제 PAD 장치
python -m gait_assistance.main live --port /dev/ttyUSB0 --duration 60
```

`offline` 실행 결과물: `stride_log.csv`, `stride_table.csv`(모든 중간 계산값),
`patient_model.json`, `results.png`(9개 패널).

`stride_log.csv`의 주요 컬럼:

```
stride_id, timestamp
gait_cluster, cluster_confidence, ood
d_patient, d_healthy, d_target, healthy_region_threshold,
    manifold_deviation, delta_healthy
estimated_swing_time, estimated_stance_time, estimated_stride_time,
    estimated_swing_ratio, estimated_stance_ratio,
    estimated_swing_stance_ratio, phase_source
swing_symmetry_ratio, stance_symmetry_ratio, swing_stance_symmetry_ratio
belt_excursion, peak_belt_velocity, trunk_acc_rms, trunk_gyro_rms
swing_ratio_error, belt_excursion_error, temporal_symmetry_error, trunk_error
biomechanical_deviation, reference_source, decision_state
raw_assist_gain, assist_gain, target_belt_length,
    motor_position, motor_velocity, motor_current
```

위상 파생 값에 `estimated_` 접두어가 붙는 이유는 아래 "보행 위상은 추정값이다"
절을 보십시오. **측정되지 않은 값은 0이 아니라 빈 칸**으로 기록됩니다.

`results.png`의 9개 패널: log-SPD PCA(healthy region 포함), `d_healthy`(임계선
포함), `d_patient`, swing ratio(정상 구간 밴드 포함), belt excursion(정상 구간
밴드 포함), `E_B`, `E_R`, assist gain, decision state 타임라인. PCA는 시각화
전용이며 어떤 거리 계산에도 쓰이지 않습니다.

실행 로그 끝에는 숫자를 어떻게 읽어야 하는지가 함께 출력됩니다.

```
[mode] baseline_stabilization
[mode] no healthy reference -> the target is this patient's own baseline. ...
[phase] BeltVelocityPhaseDetector: ESTIMATED phase from belt extension velocity
        (proximal proxy), not validated against foot contact (FSR / foot IMU pending)
[warn] no healthy reference: running in BASELINE_STABILIZATION, ...
```

설정 파일 없이도 모든 항목을 덮어쓸 수 있습니다.

```bash
python -m gait_assistance.main offline data.csv \
    --set assist.max_gain=0.6 \
    --set patient.alpha=0.5 \
    --set 'feature.channels=["belt_length","belt_velocity","gyro_z"]'
```

## 하드웨어 연동

live 경로는 상위 디렉터리의 기존 드라이버를 그대로 사용합니다. 텔레메트리는
`connect.PadController`(`PadSensorSource`), 전류 명령은 `PadMotor`,
힘-전류 변환은 `pad_external_control_lib.force_to_current_a`를 사용하며,
해당 라이브러리를 임포트할 수 없을 때를 대비해 동일한 로컬 대체 구현이 있습니다.
기본 위상 검출기는 `heelstrike_detect.py`를 포팅한 것이라 기존 튜닝값이 그대로
적용됩니다. 다만 그 튜닝값 역시 벨트 신호에 맞춘 것이지 접촉 ground truth에
맞춘 것이 아니므로, 위 "보행 위상은 추정값이다" 절의 제약이 그대로 따라옵니다.

`sensor.csv_column_map`은 기본값이 비어 있는데, 기존 수집 스크립트의 CSV 컬럼
(`host_time_s`, `belt_length`, `motor_iq_meas`, `accel_*`, `gyro_*`, ...)이
자동으로 인식되기 때문입니다.

## 보행 위상: 지금은 고정 캐이던스 placeholder

실제 swing/stance 검출 코드는 나중에 합칠 예정이므로, 기본 검출기는
`ScheduledPhaseDetector`입니다. **센서를 전혀 보지 않고 타임스탬프만으로**
규칙적인 swing/stance를 만들어 나머지 파이프라인을 끝까지 돌릴 수 있게 합니다.

```
u = (t - t0 - offset) mod period
u <  stance_duration  ->  STANCE, progress = u / stance_duration
u >= stance_duration  ->  SWING,  progress = (u - stance) / swing
```

타이밍은 전부 조정 가능합니다.

| 설정 | 기본값 | 의미 |
|---|---|---|
| `phase.scheduled_period_s` | `1.2` | stride 주기 (HS→HS, 초) |
| `phase.scheduled_swing_ratio` | `0.40` | 주기 중 swing 비율 |
| `phase.scheduled_swing_duration_s` | `0.0` | swing을 초 단위로 직접 지정 (0이면 비율 사용) |
| `phase.scheduled_offset_s` | `0.0` | 사이클 전체를 시간축에서 이동 |

```bash
python -m gait_assistance.main offline data.csv \
    --set phase.scheduled_period_s=0.9 \
    --set phase.scheduled_swing_ratio=0.35
```

`swing_duration_s()`는 stance가 항상 양수로 남도록 클램프하므로, swing이 주기를
통째로 삼키는 설정은 만들어지지 않습니다.

> **이건 시계이지 측정이 아닙니다.** 여기서 나오는 캐이던스는 설정한 값 그대로이며
> 환자의 실제 보행과 일치한다면 우연입니다. `swing_time`, `stance_time`, 모든
> ratio, 모든 symmetry는 환자를 설명하지 않습니다. 그래서 이 검출기는
> `is_placeholder = True`, `phase_source = "scheduled"`를 선언하고, 리포트에서
> "ESTIMATED"가 아니라 **"PLACEHOLDER"** 로 표시됩니다.

### 결과적으로 지금 죽어 있는 항

캐이던스가 고정이면 `swing_ratio`는 매 stride 정확히 같은 값이므로 정상 구간을
벗어날 수 없습니다. 즉 **`swing_ratio` deficit 항은 실제 검출기가 들어오기 전까지
발화하지 않습니다.** 실행 시 경고로 알려줍니다.

```
[warn] these weighted terms cannot fire until the real detector lands: swing_ratio.
       Assistance can currently only be driven by the sensor-derived terms.
```

가중치를 조용히 바꾸지는 않습니다 — 실제 검출기가 붙는 순간 그대로 살아나야 하기
때문입니다. 지금 보조를 움직일 수 있는 것은 센서에서 직접 나오는
`belt_excursion` deficit입니다.

기존 신호 기반 검출기는 그대로 남아 있어 한 줄로 되돌릴 수 있습니다.

```bash
--set phase.detector=belt_velocity     # 또는 gyro
```

## 실제 검출기를 붙이는 자리

[gait/phase_detector.py](gait_assistance/gait/phase_detector.py)가 seam입니다.
새 검출기는 세 가지만 하면 되고 **다른 모듈은 건드릴 필요가 없습니다.**

1. `PhaseDetector`를 상속해 `update()` 구현. stride를 자르는 기준인
   `GaitEvent.HEEL_STRIKE`와 stance→swing 전환의 `GaitEvent.TOE_OFF`를 내보내고,
   `_enter()` / `_register_heel_strike()`를 호출해 상속된 통계를 채웁니다.
   `PhaseResult.phase_progress`는 반드시 채우십시오 — 보조력을 swing 안 어디에
   놓을지가 이 값으로 결정되고, `None`이면 보조가 나가지 않습니다.
2. 출처 속성 선언: `phase_source`(로그 컬럼에 그대로 기록됨), `estimates_from`,
   접촉 신호로 검증했다면 `ground_truth_validated = True`와 `validated_against`.
3. 등록:

```python
from gait_assistance.gait.phase_detector import register_phase_detector
register_phase_detector("fsr", FsrPhaseDetector)
```

이후 `--set phase.detector=fsr`로 선택됩니다. 검출기별 튜닝 값은 `PhaseConfig`에
추가하면 됩니다. 같은 이름에 다른 클래스를 등록하려 하면 거부되므로, 기존 설정이
조용히 다른 것을 가리키게 되는 일은 없습니다.

`phase_source`는 검출기 → `PhaseResult` → `Stride` → `GaitMetrics` →
`stride_log.csv`까지 자동으로 흐르므로, 나중에 같은 로그 파일 안에서
`scheduled` 실행과 `fsr_contact` 실행이 구분됩니다.

## 신호 기반 검출기는 "추정값"이다

(아래는 `phase.detector`를 `belt_velocity` 또는 `gyro`로 바꿨을 때의 이야기입니다.)

두 검출기 모두 **추정된 보행 위상(estimated gait phase)** 을
내놓습니다. 어느 쪽도 발-지면 접촉을 관측하지 않고, 벨트 이동량이나 정강이
각속도라는 **근위부 대리 신호(proximal proxy)** 로부터 위상을 추론합니다.
따라서 아직 **어떤 ground truth로도 검증되지 않았습니다.**

* 위상 경계는 실제 toe-off / heel strike가 아니라 *벨트* 또는 *정강이* 이벤트의
  경계이며, 그 사이 시간 오프셋은 측정된 바 없습니다.
* 벨트 속도 검출기는 *벨트가 늘어나는* 구간 전체를 SWING으로 표시하는데, 이는
  생체역학적 유각기보다 깁니다. 즉 `swing_time`, `stance_time`, `swing_ratio`는
  **눈금이 교정되지 않은 추정치**이며, 0.38 같은 문헌 정상값과 비교하면 안 됩니다.
* 정직한 사용법은 *상대 비교*뿐입니다 — 같은 환자의 다른 스트라이드를 같은
  검출기로 본 값끼리만 비교하십시오.
* 그래서 healthy reference가 없을 때 생체역학 목표값은 인구 평균이 아니라 환자
  본인의 베이스라인을 씁니다. 0.38을 쓰면 매 스트라이드마다 swing 항이 포화됩니다.

**남은 과제:** FSR 인솔/풋 스위치 또는 발 부착 IMU로 검증해야 이 값들이 추정치가
아닌 측정치가 됩니다. 검출기 클래스는 `ground_truth_validated = False`를 명시하고
있으며, 검증된 검출기는 이 값을 True로 두고 `validated_against`에 무엇으로
검증했는지 적어야 합니다. 현재 상태는 `PhaseDetector.describe()`와 `inspect`
출력의 `phase source` 줄에서 바로 확인할 수 있습니다.

## 튜닝 참고사항

* `phase.min_belt_mm`은 관례상 음수입니다. 힐 스트라이크 조건은
  `belt_length <= min_belt_mm` 입니다. 녹화 데이터의 벨트 범위와 반드시 맞아야
  하므로, 다른 작업에 앞서 `inspect`로 몇 개의 스트라이드가 채택되는지 확인하십시오.
* `patient.baseline_strides`의 기본값은 **30**이고 권장 범위는 **30~50**입니다.
  베이스라인 하나가 z-score 통계, 로그 도메인 중심점, *그리고* k-means 분할까지
  전부 고정해야 하는데, 20 스트라이드로는 클러스터가 너무 작게 쪼개져 실루엣과
  안정성 게이트가 의미를 갖기 어렵습니다. 30 미만이면 모델에 경고가 붙습니다
  (`[warn]` 줄, `patient_model.json`의 `warnings`).
* `cluster.k_candidates`는 `[1, 2, 3, 4]`이며, K를 3으로 **강제하지 않습니다**.
  K >= 2는 세 관문을 **모두** 통과해야 채택됩니다.
  1. **최소 클러스터 크기**: 모든 클러스터가
     `max(cluster.min_cluster_size, ceil(cluster.min_cluster_fraction * n))`
     이상을 담아야 합니다. 기본값 `(3, 0.15)`로 30 스트라이드면 클러스터당 5개
     이상이 필요합니다. 절대 하한만 두면 이상치 두어 개로 K가 채택되므로,
     베이스라인이 커질수록 함께 조여지는 상대 조건을 같이 겁니다. 어떤 K가
     `k * required > n`이라 애초에 불가능하면 k-means를 돌리지도 않습니다.
  2. 실루엣 >= `min_silhouette`
  3. 부트스트랩 안정성 >= `min_stability`

  하나라도 못 넘으면 K = 1이 반환됩니다. 어떤 K가 왜 떨어졌는지는 요약 JSON의
  `k_selection` 배열에 후보별로 `silhouette`, `stability`, `min_cluster_size`,
  `required_cluster_size`, `accepted`, `reason`이 그대로 남습니다.

### 실루엣 값을 해석하지 마십시오

실루엣과 안정성은 **분할의 입장 조건**일 뿐, 보행의 품질 지표가 아닙니다.
실루엣이 높다는 것은 찾아낸 클러스터들이 기하학적으로 잘 분리돼 있다는 뜻이고,
병적 보행에서 값이 높으면 그저 **그 병적 패턴이 일관적이라는** 뜻입니다.

그래서 이 패키지는 실루엣 값을 "좋다/나쁘다"로 자동 해석하는 코드를 두지 않습니다.
`summary()`도 CLI 리포트도 측정값과 채택/기각 사유만 출력하고 등급을 매기지
않습니다. 이 숫자로 환자 상태나 세션 품질을 판단하지 마십시오.

## 테스트

```bash
python -m pytest gait_assistance/tests -q
```

`test_assistance.py`가 추가로 검증하는 항목: healthy centroid에서 거리 0,
region 내부 `E_R = 0` / 외부 `E_R > 0`, 세 가지 임계값 방법, reference 저장/로드
왕복, 정상 구간 내부 error 0 / 하한 미만 deficit / 방향성 처리, 미측정 값의
error가 `None`인 것, `swing + stance ≈ stride`와 `swing_ratio + stance_ratio ≈ 1`,
`swing_stance_ratio` 정확성, 좌우 동일 시 ratio 1 / 반대측 없으면 `None`,
`(E_R, E_B)` 네 조합의 게인 거동, OOD에서 강한 보조 금지, persistence(1 stride
이상은 무시, 연속 조건 충족 시 변경, 복귀도 동일), target mode 3종.

`test_scheduled_phase.py`가 검증하는 항목: 설정한 주기대로 heel strike가 발생,
설정한 비율대로 toe-off가 사이클을 분할, 명시적 swing 지속시간이 비율을 덮어씀,
offset이 사이클을 이동, swing이 주기를 삼키지 않음, segmenter가 주기당 stride
하나를 자름, `phase_source`가 stride와 metrics까지 전달됨, 새 검출기 등록/선택과
중복 이름 거부, **stance에서 보조력 0 / swing에서만 벨트 수축**, progress가 없으면
보조하지 않음.

기존 검증 항목: 공분산의 대칭성과 양의 정부호성, `matrix_log`의 대칭성,
`d_LE(C, C) = 0`과 그 대칭성, 벡터화의 등거리성, 최근접 중심 할당,
K 선택(비구조적 데이터에서 K = 1인 경우 포함), OOD 검출,
`gain <= max_gain`, `|gain_t - gain_{t-1}| <= max_gain_delta`, NaN 안전성,
모터 포화, 스트라이드 리샘플링 출력 길이,
보조 모드 판정과 두 편차의 분리(healthy reference 없을 때 `e_healthy`가 NaN으로
유지되는지), 상대 최소 클러스터 크기 조건.

## 요구 사항

`numpy`, `scipy`, `pandas`, `scikit-learn`, 그림 생성을 위한 `matplotlib`,
테스트를 위한 `pytest`. 그 외에는 없습니다.
