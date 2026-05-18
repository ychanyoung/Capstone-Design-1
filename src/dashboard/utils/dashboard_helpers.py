"""
Shared Dashboard Utility Functions.

Provides formatting, validation, configuration extraction, chart helpers,
risk classification, and page routing utilities used across all Streamlit
dashboard pages.

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import datetime as dt_module
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# =========================================================================
# Page definitions
# =========================================================================

PAGES = [
    "Demo",
    "Overview",
    "Churn Analytics",
    "Model Performance",
    "Customer Segmentation",
    "Cohort Analysis",
    "Budget Optimization",
    "A/B Testing",
    "Survival Analysis",
    "Model Monitoring",
    "Recommendations",
    "CLV Prediction",
    "Uplift Modeling",
    "CLV & Retention Campaign",
    "Real-Time Scoring",
    "MLflow Experiments",
    "System Health",
]

# =========================================================================
# Internationalization (i18n) — Korean/English toggle
# =========================================================================
# Currently translated: sidebar navigation labels + a small set of
# high-visibility labels (Navigation title, Select Page legend, sidebar
# section headers, common KPI labels). Pages keep English chart content
# for now; toggle affects shell + navigation.

I18N_KO = {
    # --- Demo (시연) page -----------------------------------------------
    "Demo": "시연",
    "Demo: Coupon Recommendation Impact on CLV": "시연: 쿠폰 추천이 CLV에 미치는 영향",
    "Move the budget slider to see how our coupon recommendations convert spend into customer lifetime value in real time.": "예산 슬라이더를 움직여 쿠폰 추천이 예산을 어떻게 고객 생애 가치로 전환하는지 실시간으로 확인하세요.",
    "Move the budget slider to see how our coupon recommendations convert spend into customer lifetime value in real time. Numbers come from the same LP solution shown on the Budget Optimization page.": "예산 슬라이더를 움직여 쿠폰 추천이 예산을 어떻게 고객 생애 가치로 전환하는지 실시간으로 확인하세요. 수치는 예산 최적화 페이지와 동일한 LP 솔루션에서 가져옵니다.",
    "No LP solution available yet. Run the pipeline first: `docker compose up pipeline`": "아직 LP 솔루션이 없습니다. 먼저 파이프라인을 실행하세요: `docker compose up pipeline`",
    "LP artefact is present but contains no allocated budget. Rerun the pipeline with a non-zero budget.": "LP 산출물은 있지만 배정된 예산이 0입니다. 0이 아닌 예산으로 파이프라인을 다시 실행하세요.",
    "Slider range: 0 to the 200 % LP what-if scenario. The headline Budget Optimization number lives at the 100 % anchor.": "슬라이더 범위: 0부터 LP 200% what-if 시나리오까지. 예산 최적화 페이지의 헤드라인 수치는 100% 앵커에 위치합니다.",
    "Anchor point: at the LP's nominal budget ": "앵커: LP 기본 예산 ",
    "the saved revenue equals ": "에서 절감된 매출은 ",
    "— this is the headline figure on the Budget Optimization page.": " 입니다 — 이는 예산 최적화 페이지의 헤드라인 수치와 동일합니다.",
    "LP @ 50%": "LP @ 50%",
    "LP @ 100%": "LP @ 100%",
    "LP @ 200%": "LP @ 200%",
    "LP what-if anchors": "LP what-if 앵커",
    "customers": "명",
    "Budget → CLV / Net Gain (LP-grounded)": "예산 → CLV / 순이득 (LP 기반)",
    "- **Baseline CLV** = Σ `clv × (1 − churn_prob)` across all customers in `budget_optimization.csv`.\n- **Selection rule** = ROI-per-won greedy on the LP solution. Customers the LP has chosen are ranked by `expected_revenue_saved_krw / allocated_budget` and treated in order until the budget is exhausted.\n- **At slider = LP nominal budget** the greedy selection trivially equals the full LP allocation, so the headline **CLV after recommendations** and **Net gain** match the Budget Optimization page exactly.\n- **Above the LP cap** the curve linearly interpolates between the 100 % and 200 % what-if scenarios from `budget_whatif.csv` — these are independent LP reruns shown as orange diamonds on the chart.\n- **Net gain** = Σ `expected_revenue_saved_krw` − Σ `allocated_budget` of treated customers.": "- **기준 CLV** = `budget_optimization.csv`의 모든 고객에 대한 Σ `clv × (1 − churn_prob)`.\n- **선택 규칙** = LP 솔루션에 대한 원당 ROI 그리디. LP가 선택한 고객을 `expected_revenue_saved_krw / allocated_budget`로 정렬해 예산이 소진될 때까지 처치.\n- **슬라이더 = LP 기본 예산**일 때 그리디 선택은 자명하게 전체 LP 배정과 같으므로, 헤드라인 **추천 후 CLV**와 **순이득**이 예산 최적화 페이지와 정확히 일치합니다.\n- **LP cap 초과 구간**에서는 `budget_whatif.csv`의 100% / 200% what-if 시나리오 사이를 선형 보간 — 이들은 독립적인 LP 재실행이며 차트의 주황색 다이아몬드로 표시됩니다.\n- **순이득** = 처치된 고객의 Σ `expected_revenue_saved_krw` − Σ `allocated_budget`.",
    "Adjust budget": "예산 조정",
    "Budget (KRW)": "예산 (원)",
    "Baseline CLV (no coupon)": "기준 CLV (쿠폰 없음)",
    "Uncouponed CLV (baseline)": "쿠폰 미적용 CLV",
    "Uniform-Treatment CLV (avg coupon for all)": "전체 적용 CLV (모든 고객 평균 쿠폰)",
    "Segment-level Budget Allocation": "세그먼트별 예산 배정",
    "Customers": "고객 수",
    "Run pipeline to compute": "파이프라인 실행 필요",
    "CLV after recommendations": "추천 후 CLV",
    "Net gain (ΔCLV − coupon cost)": "순이득 (ΔCLV − 쿠폰비용)",
    "Customers treated": "처치 고객 수",
    "spent": "지출",
    "Baseline CLV": "기준 CLV",
    "Current budget": "현재 예산",
    "Budget → CLV / Net Gain": "예산 → CLV / 순이득",
    "CLV after recommendations (KRW)": "추천 후 CLV (원)",
    "Net gain (KRW)": "순이득 (원)",
    "Net gain": "순이득",
    "How is this computed?": "어떻게 계산되나요?",
    "- **Baseline CLV** = Σ `clv × (1 − churn_probability)` across all customers.\n- **Coupon selection** = ROI-per-won greedy: customers with the highest `expected_revenue_saved / estimated_cost` are treated first until the budget is exhausted.\n- **CLV after recommendations** = Baseline + Σ `expected_revenue_saved` of treated customers.\n- **Net gain** = Σ `expected_revenue_saved` − Σ `estimated_cost` of treated customers.": "- **기준 CLV** = 전체 고객에 대한 Σ `clv × (1 − churn_probability)`.\n- **쿠폰 선택** = 원당 ROI 그리디 방식: `expected_revenue_saved / estimated_cost` 가 높은 고객 순으로 예산 한도까지 처치.\n- **추천 후 CLV** = 기준 CLV + 처치된 고객의 Σ `expected_revenue_saved`.\n- **순이득** = 처치된 고객의 Σ `expected_revenue_saved` − Σ `estimated_cost`.",
    "No recommendations available yet. Run the pipeline first: `docker compose up pipeline`": "아직 추천 결과가 없습니다. 먼저 파이프라인을 실행하세요: `docker compose up pipeline`",
    "Data loader unavailable.": "데이터 로더를 사용할 수 없습니다.",
    "Failed to load recommendations:": "추천 데이터를 불러오지 못했습니다:",
    "(Authoritative health status comes from the Drift Detection Overview banner.)": "(권위 있는 헬스 상태는 Drift 탐지 개요 배너에서 가져옵니다.)",
    "1. Customer Lifetime Value Overview": "1. 고객 생애 가치 개요",
    "2. Uplift Modeling & Treatment Effectiveness": "2. 업리프트 모델링 및 처치 효과성",
    "3. Budget Optimization Outcomes": "3. 예산 최적화 결과",
    "4. Campaign ROI Metrics": "4. 캠페인 ROI 지표",
    "50% Survival (Median)": "50% 생존 (중앙값)",
    "80% Power": "80% 검정력",
    "80% threshold": "80% 임계값",
    "A/B Testing": "A/B 테스트",
    "A/B Testing Results": "A/B 테스트 결과",
    "AI-driven retention recommendations based on churn risk, CLV, uplift scores, and customer segment affinity.": "이탈 위험, CLV, 업리프트 점수, 고객 세그먼트 친화도에 기반한 AI 기반 유지 추천.",
    "AUC": "AUC",
    "AUC by Model Type": "모델 유형별 AUC",
    "AUC differences between models are within statistical noise (Δ={auc_margin:.4f}, no DeLong / significance test performed). Treat ranking as indicative, not definitive.": "모델 간 AUC 차이는 통계적 노이즈 범위 내입니다 (Δ={auc_margin:.4f}, DeLong / 유의성 검정 미실행). 순위는 참고용으로 다루세요.",
    "AUC per Training Second (Efficiency)": "학습 1초당 AUC (효율성)",
    "AUC vs Training Time": "AUC vs 학습 시간",
    "AUC vs Training Time Trade-off": "AUC vs 학습 시간 trade-off",
    "Absolute Effect (Churn Reduction)": "절대 효과 (이탈 감소)",
    "Action EV (uplift × CLV)": "액션 EV (업리프트 × CLV)",
    "Action Type": "액션 유형",
    "Action Type Distribution": "액션 유형 분포",
    "Actionable Recommendations": "실행 가능 추천",
    "Actual": "실제",
    "Actual No": "실제 No",
    "Actual Yes": "실제 Yes",
    "Adjust campaign cost assumptions (1.0 = baseline)": "캠페인 비용 가정 조정 (1.0 = 기준)",
    "Adjust uplift effectiveness assumptions (1.0 = baseline)": "업리프트 효과 가정 조정 (1.0 = 기준)",
    "Aggregate ROI = total revenue saved / total budget allocated. ": "집계 ROI = 총 절감 매출 / 총 배정 예산. ",
    "Aggregated as int(sum(per-segment retained)). Matches the Baseline / Current Selection rows of the What-If Scenario Comparison table by construction (iter11 reconciliation).": "세그먼트별 유지 고객 수의 합으로 집계됨. What-If 시나리오 비교 표의 Baseline/Current Selection 행과 정의상 일치.",
    "All": "전체",
    "All KPIs are simulator-generated.": "모든 KPI는 시뮬레이터 생성입니다.",
    "All Metrics by Model": "모델별 전체 지표",
    "All Systems Operational": "모든 시스템 정상",
    "Allocated": "배정",
    "Allocated Budget": "배정 예산",
    "Allocation Distribution": "배정 분포",
    "Allocation Proportions": "배정 비율",
    "Allocation Summary": "배정 요약",
    "Allocation is limited by segment-size cap (binding constraint: segment_size - only a small population of high-value persuadable customers exists, so the LP cannot scale spend further on this segment regardless of its per-unit ROI).": "세그먼트 크기 cap에 의해 배정이 제한됨 (활성 제약: segment_size - 고가치 persuadable 고객 모집단이 작아 단위당 ROI와 무관하게 LP가 추가 지출 확대 불가).",
    "Artifacts": "Artifact",
    "At-Risk CLV": "위험 CLV",
    "At-Risk CLV %": "위험 CLV %",
    "At-Risk Revenue (churn prob > 50%)": "위험 매출 (이탈 확률 > 50%)",
    "Average CLV": "평균 CLV",
    "Average CLV by Segment": "세그먼트별 평균 CLV",
    "Average CLV by Segment (n>=5 only)": "세그먼트별 평균 CLV (n>=5만)",
    "Average Churn Probability by Segment": "세그먼트별 평균 이탈 확률",
    "Average Churn Risk by Segment": "세그먼트별 평균 이탈 위험",
    "Average Expected Uplift by Segment": "세그먼트별 평균 예상 업리프트",
    "Average Predicted Uplift by Treated Action": "처치 액션별 평균 예측 업리프트",
    "Average Retention": "평균 유지율",
    "Average Retention Curve": "평균 유지 곡선",
    "Average Retention Rate Across All Cohorts": "전체 코호트 평균 유지율",
    "Average Scoring Latency": "평균 채점 지연시간",
    "Average Survival Probability by Uplift Segment": "업리프트 세그먼트별 평균 생존 확률",
    "Average Survival Probability by Uplift Segment (Cox PH-derived)": "업리프트 세그먼트별 평균 생존 확률 (Cox PH 도출)",
    "Average Uplift Score by Segment": "세그먼트별 평균 업리프트 점수",
    "Avg CLV": "평균 CLV",
    "Avg Churn Prob": "평균 이탈 확률",
    "Avg Deepest-Observed Retention": "관측된 최장 유지율 평균",
    "Average Survival Probability at 1 Year by Behavioral Segment (Cox PH)": "행동 세그먼트별 1년 평균 생존 확률 (Cox PH)",
    "Average Survival Probability by Behavioral Segment": "행동 세그먼트별 평균 생존 확률",
    "Bars show mean Cox PH-derived survival probability at t=365 days for each behavioral segment (same 6 segments as the Kaplan-Meier curves above). Earlier versions of this chart used t=90 days, at which point every segment was still near the ceiling and the bars looked artificially uniform. For uplift-segment analysis see Page 11 (Uplift Modeling).": "각 행동 세그먼트별 t=365일 시점의 Cox PH 평균 생존 확률을 보여줍니다 (위 Kaplan-Meier 곡선과 동일한 6개 세그먼트). 이전 버전은 t=90일을 사용했는데 그 시점에는 모든 세그먼트가 천장 근처라 막대가 균일해 보였습니다. 업리프트 세그먼트 분석은 Page 11(업리프트 모델링)을 참고하세요.",
    "Ground-truth churn rate of the generated customer simulator (label-based, PRD target 15-25%). This differs from the model's mean predicted probability shown on the Churn Analytics page.": "생성된 고객 시뮬레이터의 라벨 기반 실제 이탈률(PRD 목표 15-25%). 이탈 분석 페이지의 모델 평균 예측 확률과 다릅니다.",
    "Ground-truth churn rate of the generated customer simulator (label-based, PRD target 15-25%).": "생성된 고객 시뮬레이터의 라벨 기반 실제 이탈률(PRD 목표 15-25%).",
    "Mean Predicted Probability": "평균 예측 확률",
    "Mean of the model's predicted churn probability across all customers. Right-skewed distribution means this typically exceeds the label rate.": "전체 고객에 대한 모델의 평균 예측 이탈 확률. 우편향 분포 때문에 보통 라벨 이탈률보다 높습니다.",
    "Median Predicted Probability": "중앙값 예측 확률",
    "Simulator Churn Rate": "시뮬레이터 이탈률",
    "Avg Error Rate": "평균 오류율",
    "Avg Expected Uplift": "평균 기대 업리프트",
    "Avg Final Retention": "평균 최종 유지율",
    "Avg Latency": "평균 지연시간",
    "Avg Latency (ms)": "평균 지연시간 (ms)",
    "Avg Lift": "평균 lift",
    "Avg Period-1 Retention": "평균 1기 유지율",
    "Avg Predicted Uplift (all customers)": "평균 예측 업리프트 (전체 고객)",
    "Avg Predicted Uplift (treated)": "평균 예측 업리프트 (처치 고객)",
    "Avg Priority Score": "평균 우선순위 점수",
    "Avg ROI": "평균 ROI",
    "Avg ROI (x)": "평균 ROI (x)",
    "Avg Requests/min": "평균 요청/분",
    "Avg Throughput": "평균 처리량",
    "Avg Treated Uplift": "평균 처치 업리프트",
    "Avg Uplift": "평균 업리프트",
    "Avg Uplift Score": "평균 업리프트 점수",
    "Avg per-segment ROI": "세그먼트별 평균 ROI",
    "Based on 100 new enrollments per day": "하루 100명 신규 가입 기준",
    "Baseline Churn Rate": "기준 이탈률",
    "Best AUC": "최고 AUC",
    "Best Experiment": "최고 실험",
    "Best Model": "최고 모델",
    "Best model": "최고 모델",
    "Best model by AUC": "AUC 기준 최고 모델",
    "Bottom": "하위",
    "Bottom 10 Customers by CLV": "CLV 기준 하위 10명 고객",
    "Budget": "예산",
    "Budget Allocated": "배정 예산",
    "Budget Allocation by Channel": "채널별 예산 배정",
    "Budget Allocation by Segment": "세그먼트별 예산 배정",
    "Budget Constraints & Scenario Parameters": "예산 제약 및 시나리오 파라미터",
    "Budget Efficiency: Spend vs Revenue Saved": "예산 효율: 지출 vs 절감 매출",
    "Budget Optimization": "예산 최적화",
    "Budget Share by Segment": "세그먼트별 예산 비중",
    "Budget Sweep Analysis": "예산 스윕 분석",
    "Budget Sweep: Retained Customers & Revenue Saved": "예산 스윕: 유지 고객 및 절감 매출",
    "C": "C",
    "CLV & Retention Campaign": "CLV 및 유지 캠페인",
    "CLV Distribution": "CLV 분포",
    "CLV Distribution by Risk Level": "위험 등급별 CLV 분포",
    "CLV Distribution by Segment": "세그먼트별 CLV 분포",
    "CLV Percentile Analysis": "CLV 백분위 분석",
    "CLV Prediction": "CLV 예측",
    "CLV Std Dev": "CLV 표준편차",
    "CLV Tier Classification": "CLV 등급 분류",
    "CLV Tier Distribution": "CLV 등급 분포",
    "CLV by Percentile": "백분위별 CLV",
    "CLV by Segment": "세그먼트별 CLV",
    "CLV vs Churn Probability (High CLV + High Churn = Priority)": "CLV vs 이탈 확률 (고 CLV + 고 이탈 = 우선순위)",
    "CLV vs Churn Risk": "CLV vs 이탈 위험",
    "Campaign Effectiveness by Segment": "세그먼트별 캠페인 효과성",
    "Change in Retention (%)": "유지율 변화 (%)",
    "Channel": "채널",
    "Channel Cost & ROI Details": "채널 비용 및 ROI 세부사항",
    "Channel Efficiency Frontier": "채널 효율 frontier",
    "Channel configuration not found in config. Add budget.channels to simulator_config.yaml for multi-channel allocation views.": "config에서 채널 설정을 찾을 수 없습니다. 다채널 배정 보기를 위해 simulator_config.yaml에 budget.channels를 추가하세요.",
    "Channel-Level Cost Breakdown": "채널 단위 비용 분해",
    "Chart suppressed to avoid a time-anchor split with the drift trend below. Restart the scoring pipeline to refresh.": "아래 drift 추세와의 시간 앵커 분리를 피하기 위해 차트가 숨겨졌습니다. 새로 고침하려면 채점 파이프라인을 재시작하세요.",
    "Churn Analytics": "이탈 분석",
    "Churn Definition": "이탈 정의",
    "Churn Drivers Correlation": "이탈 요인 상관관계",
    "Churn Event Rate by Uplift Segment (label-leak — see warning)": "업리프트 세그먼트별 이탈 이벤트 비율 (라벨 누설 — 경고 참조)",
    "Churn Prediction Analytics": "이탈 예측 분석",
    "Churn Prediction Features": "이탈 예측 피처",
    "Churn Prediction Overview": "이탈 예측 개요",
    "Churn Probability": "이탈 확률",
    "Churn Probability Density by Segment": "세그먼트별 이탈 확률 밀도",
    "Churn Probability Distribution": "이탈 확률 분포",
    "Churn Probability Distribution (Recent Scores)": "이탈 확률 분포 (최근 점수)",
    "Churn Probability Distribution by Risk Level": "위험 등급별 이탈 확률 분포",
    "Churn Probability Distribution by Segment": "세그먼트별 이탈 확률 분포",
    "Churn Probability by Risk Level": "위험 등급별 이탈 확률",
    "Churn Probability vs Expected Uplift": "이탈 확률 vs 예상 업리프트",
    "Churn Probability vs Predicted CLV": "이탈 확률 vs 예측 CLV",
    "Churn Rate": "이탈률",
    "Churn Rate by Customer Segment": "고객 세그먼트별 이탈률",
    "Churn Risk Score": "이탈 위험 점수",
    "Churn Risk Score Distribution": "이탈 위험 점수 분포",
    "Churn Risk Summary": "이탈 위험 요약",
    "Churn Risk vs Customer Lifetime Value": "이탈 위험 vs 고객 생애 가치",
    "Churn probability threshold": "이탈 확률 임계값",
    "Cohort": "코호트",
    "Cohort Analysis": "코호트 분석",
    "Cohort Overview": "코호트 개요",
    "Cohort Sizes": "코호트 크기",
    "Compare budget optimization outcomes across different budget levels and parameter assumptions.": "다양한 예산 수준과 파라미터 가정에 따른 예산 최적화 결과를 비교합니다.",
    "Computed only over observed cells (cohorts whose follow-up window has actually elapsed). Zero-filled future cells are excluded — closes iter9 P04 #17.": "추적 기간이 실제로 경과한 코호트의 관측 셀만으로 계산. 0으로 채워진 미래 셀은 제외.",
    "Configuration": "설정",
    "Confusion Matrices": "혼동 행렬",
    "Connected": "연결됨",
    "Connected to MLflow tracking server": "MLflow 추적 서버에 연결됨",
    "Consider tightening the MDE target or running a longer experiment with cohort rotation.": "MDE 목표를 더 엄격하게 잡거나, 코호트 순환과 함께 실험 기간을 늘리는 것을 고려하세요.",
    "Consumer Group": "Consumer 그룹",
    "Consumer Groups": "Consumer 그룹",
    "Consumers": "Consumer",
    "Control": "대조군",
    "Control Churn": "대조군 이탈",
    "Cost": "비용",
    "Cost Multiplier": "비용 배수",
    "Cost per Action": "액션당 비용",
    "Cost per Retained Customer": "유지 고객당 비용",
    "Cost vs Expected Revenue Saved by Segment": "세그먼트별 비용 vs 예상 절감 매출",
    "Cost vs Revenue Saved per Customer": "고객당 비용 vs 절감 매출",
    "Cost-Benefit Analysis": "비용 편익 분석",
    "Cost-Effectiveness Analysis": "비용 효과 분석",
    "Count": "개수",
    "Count of customers whose predicted churn probability exceeds 50%. This is a prediction-derived label and matches the High Risk count on Page 01 by construction — it is NOT an observed event count (iter9 audit P07 #19).": "예측 이탈 확률 50% 초과 고객 수. 이는 예측 도출 라벨이며 페이지 01의 High Risk 카운트와 정의상 일치 — 관측 이벤트 카운트가 아닙니다.",
    "Critical": "위급",
    "Critical (>75%)": "위급 (>75%)",
    "Cross-Experiment Comparison": "실험 간 비교",
    "Crosswalk note: this chart groups by the **uplift taxonomy** (high/mid/low_value × persuadable/sure_thing/lost_cause/sleeping_dog), while the Kaplan-Meier curves and Daily Hazard chart above use the **behavioral taxonomy** (vip_loyal, regular_loyal, bargain_hunter, explorer, dormant, new_customer — same as Page 03). The two taxonomies are not interchangeable; do not compare bars across the two charts.": "크로스워크 참고: 이 차트는 **업리프트 분류 체계**(high/mid/low_value × persuadable/sure_thing/lost_cause/sleeping_dog)로 그룹화되며, 위의 Kaplan-Meier 곡선과 일별 위험률 차트는 **행동 분류 체계**(vip_loyal, regular_loyal, bargain_hunter, explorer, dormant, new_customer — 페이지 03과 동일)를 사용합니다. 두 분류 체계는 호환되지 않으므로, 두 차트 간 막대를 비교하지 마세요.",
    "Cumulative Feature Importance": "누적 피처 중요도",
    "Cumulative Importance %": "누적 중요도 %",
    "Cumulative Importance (%)": "누적 중요도 (%)",
    "Cumulative Uplift Curve (Qini-style)": "누적 업리프트 곡선 (Qini 스타일)",
    "Current Drift Status": "현재 Drift 상태",
    "Current Model Performance": "현재 모델 성능",
    "Current Status": "현재 상태",
    "Customer": "고객",
    "Customer Count": "고객 수",
    "Customer Duration Distribution by Segment": "세그먼트별 고객 기간 분포",
    "Customer Lifetime Duration Distribution": "고객 생애 기간 분포",
    "Customer Lifetime Value Distribution": "고객 생애 가치 분포",
    "Customer Response Classification": "고객 반응 분류",
    "Customer Retention by Cohort (%)": "코호트별 고객 유지율 (%)",
    "Customer Risk Level Distribution": "고객 위험 등급 분포",
    "Customer Risk Levels": "고객 위험 등급",
    "Customer Segment": "고객 세그먼트",
    "Customer Segment Distribution": "고객 세그먼트 분포",
    "Customer Segment Overview": "고객 세그먼트 개요",
    "Customer Segmentation": "고객 세그멘테이션",
    "Customers Retained": "유지된 고객 수",
    "Customers per Segment": "세그먼트별 고객 수",
    "D": "D",
    "DL Model": "DL 모델",
    "DL Model AUC": "DL 모델 AUC",
    "DL Weight": "DL 가중치",
    "Daily Hazard Rate by Segment": "세그먼트별 일별 위험률",
    "Date": "날짜",
    "Days Since First Purchase": "첫 구매 이후 일수",
    "Days Since Purchase": "마지막 구매 후 일수",
    "Default": "기본값",
    "Definitions are driven by the segments actually emitted by the runtime segmenter (6 names). iter9/iter10 audits flagged that the previous config-driven table listed 8 names — including 4 (loyal_customer, potential_loyalist, at_risk, hibernating) that do not appear in the charts above — and omitted regular_loyal and dormant (the headline highest-risk segment).": "정의는 런타임 세그멘터가 실제로 출력하는 세그먼트(6개 이름)에서 가져옵니다. iter9/iter10 감사에서 이전의 config 기반 표가 위 차트에 나타나지 않는 4개(loyal_customer, potential_loyalist, at_risk, hibernating)를 포함한 8개를 나열하고 regular_loyal과 dormant(헤드라인 최고 위험 세그먼트)를 누락한 것이 표시되었습니다.",
    "Degraded — Investigate Subsystems": "성능 저하 — 서브시스템 점검 필요",
    "Detailed Offer Recommendations": "상세 오퍼 추천",
    "Detailed Retention Offers": "상세 유지 오퍼",
    "Distribution of Churn Probabilities": "이탈 확률 분포",
    "Distribution of Churn Risk Scores with Threshold Boundaries": "임계값 경계가 있는 이탈 위험 점수 분포",
    "Distribution of Customer Durations": "고객 기간 분포",
    "Distribution of Expected Uplift Scores": "예상 업리프트 점수 분포",
    "Distribution of Uplift Scores": "업리프트 점수 분포",
    "Drift Alert Timeline": "Drift 알림 타임라인",
    "Drift Alerts Over Time": "시간에 따른 Drift 알림",
    "Drift Detection (KS)": "Drift 탐지 (KS)",
    "Drift Detection (PSI)": "Drift 탐지 (PSI)",
    "Drift Detection Log": "Drift 탐지 로그",
    "Drift Detection Overview": "Drift 탐지 개요",
    "Drift watchlist:": "Drift 감시 목록:",
    "Duration (days)": "기간 (일)",
    "Duration Distribution by Segment": "세그먼트별 기간 분포",
    "E": "E",
    "Effect Size": "효과 크기",
    "Effect Size & 95% CI": "효과 크기 및 95% CI",
    "Efficiency Frontier: Cost vs ROI": "효율 frontier: 비용 vs ROI",
    "English": "English",
    "Ensemble AUC": "앙상블 AUC",
    "Ensemble Configuration": "앙상블 구성",
    "Ensemble Improvement Over Individual Models": "개별 모델 대비 앙상블 개선",
    "Ensemble Weight Distribution": "앙상블 가중치 분포",
    "Ensemble Weights": "앙상블 가중치",
    "Epochs vs AUC (size = training time)": "Epoch vs AUC (크기 = 학습 시간)",
    "Error Rate": "오류율",
    "Error rate": "오류율",
    "Est. Revenue Saved": "예상 절감 매출",
    "Estimate required sample sizes and statistical power for planning future A/B experiments.": "향후 A/B 실험 계획에 필요한 샘플 크기와 통계적 검정력을 추정합니다.",
    "Estimated Cost": "추정 비용",
    "Estimated Hazard Rate by Behavioral Segment": "행동 세그먼트별 추정 위험률",
    "Event Rate": "이벤트 비율",
    "Event Rate by Uplift Segment": "업리프트 세그먼트별 이벤트 비율",
    "Events Observed (Churn)": "관측 이벤트 (이탈)",
    "Expected Duration (days)": "예상 기간(일)",
    "Expected ROI by Customer Segment": "고객 세그먼트별 예상 ROI",
    "Expected Retained": "예상 유지 고객 수",
    "Expected Retained Customers by Scenario": "시나리오별 예상 유지 고객",
    "Expected Retention Uplift per Customer": "고객당 예상 유지 업리프트",
    "Expected Revenue Saved": "예상 절감 매출",
    "Expected Revenue Saved by Segment": "세그먼트별 예상 절감 매출",
    "Expected Uplift": "예상 업리프트",
    "Expected Uplift Analysis": "예상 업리프트 분석",
    "Expected Uplift Distribution": "예상 업리프트 분포",
    "Expected Uplift by Customer": "고객별 예상 업리프트",
    "Expected churn rate without treatment": "처치 없을 때의 예상 이탈률",
    "Experiment": "실험",
    "Experiment Run History": "실험 실행 이력",
    "Experiment Timeline": "실험 타임라인",
    "Experiments": "실험",
    "Experiments cached": "캐시된 실험",
    "F1": "F1",
    "F1 Score": "F1 점수",
    "FAILED": "실패",
    "False Positive Rate": "위양성률 (FPR)",
    "Feasibility check: some MDE rows below require more participants than the available customer pool": "타당성 점검: 아래 MDE 행 중 일부는 가용 고객 풀보다 많은 참가자를 요구합니다",
    "Feature": "피처",
    "Feature Correlation Matrix": "피처 상관 행렬",
    "Feature Importance": "피처 중요도",
    "Feature Importance Analysis": "피처 중요도 분석",
    "Feature Importance Scores": "피처 중요도 점수",
    "Filter by Action Type": "액션 유형 필터",
    "Fraction of customers with predicted churn prob >50%.": "예측 이탈 확률 50% 초과 고객 비율.",
    "Full Configuration": "전체 설정",
    "Group": "그룹",
    "Group-size validation": "그룹 크기 검증",
    "Handles real-time scoring requests": "실시간 채점 요청 처리",
    "High": "고",
    "High Priority": "고우선순위",
    "High Risk": "고위험",
    "High Risk (>50%)": "고위험 (>50%)",
    "High Risk Customers": "고위험 고객",
    "High/Critical": "고/위급",
    "High/Critical Risk": "고/위급 위험",
    "Highest Risk Segment": "최고 위험 세그먼트",
    "Histogram bin width: 0.02 (50 bins across [0, 1]) — consistent with the Churn Analytics page so the leftmost-bin counts can be reconciled across pages.": "히스토그램 빈 너비: 0.02 (50개 구간, [0, 1] 범위) — 이탈 분석 페이지와 같은 기준이라 첫 구간 카운트를 페이지 간 비교할 수 있습니다.",
    "Historical": "과거 데이터",
    "Hyperparameter Analysis": "하이퍼파라미터 분석",
    "Importance": "중요도",
    "Importance Score": "중요도 점수",
    "Individual Customer Lookup": "개별 고객 조회",
    "Insufficient history": "이력 부족",
    "KS Statistic Trend (Kolmogorov-Smirnov)": "KS 통계 추세 (Kolmogorov-Smirnov)",
    "Kaplan-Meier Survival Curves by Customer Segment": "고객 세그먼트별 Kaplan-Meier 생존 곡선",
    "Kaplan-Meier Survival Curves by Segment": "세그먼트별 Kaplan-Meier 생존 곡선",
    "Korean": "한국어",
    "Language": "언어",
    "Last checked": "마지막 점검",
    "Last refresh": "마지막 새로고침",
    "Last update": "마지막 업데이트",
    "Latency (ms)": "지연시간 (ms)",
    "Latest Alert Level": "최신 알림 레벨",
    "Latest KS": "최신 KS",
    "Latest PSI": "최신 PSI",
    "Learning Rate vs AUC": "학습률 vs AUC",
    "Length": "길이",
    "Lift": "Lift",
    "Limited cohort window — only": "제한된 코호트 윈도우 — 단지",
    "Lost Cause": "Lost Cause",
    "Low": "저",
    "Low/Medium": "저/중",
    "MDE Sensitivity Analysis": "MDE 민감도 분석",
    "ML Model": "ML 모델",
    "ML Model AUC": "ML 모델 AUC",
    "ML Pipeline": "ML 파이프라인",
    "ML Weight": "ML 가중치",
    "MLflow Configuration": "MLflow 설정",
    "MLflow Experiment Runs": "MLflow 실험 실행",
    "MLflow Experiment Tracking": "MLflow 실험 추적",
    "MLflow Experiments": "MLflow 실험",
    "MLflow Run Performance Comparison": "MLflow 실행 성능 비교",
    "MLflow Tracking": "MLflow 추적",
    "MLflow tracking server not available ({err}) — showing cached experiment data from artifacts.": "MLflow 추적 서버 사용 불가 ({err}) — artifact의 캐시된 실험 데이터를 표시합니다.",
    "Max Uplift": "최대 업리프트",
    "Mean": "평균",
    "Mean Churn Probability Over Time": "시간에 따른 평균 이탈 확률",
    "Mean KS Statistic": "평균 KS 통계",
    "Mean KS Statistic Over Time": "시간에 따른 평균 KS 통계",
    "Mean PSI": "평균 PSI",
    "Mean PSI Over Time": "시간에 따른 평균 PSI",
    "Mean of segment ROIs": "세그먼트별 ROI 평균",
    "Median": "중앙값",
    "Median CLV": "CLV 중앙값",
    "Median Churn Prob": "이탈 확률 중앙값",
    "Median Duration": "중앙 기간",
    "Median Duration *": "중앙 기간 *",
    "Median Survival Time by Segment": "세그먼트별 중앙 생존 시간",
    "Medium": "중",
    "Medium/High": "중/고",
    "Metric": "지표",
    "Metric Comparison Across Runs": "실행 간 지표 비교",
    "Metrics Comparison Chart": "지표 비교 차트",
    "Minimum Detectable Effect (MDE)": "최소 감지 효과 (MDE)",
    "Minimum Priority Score": "최소 우선순위 점수",
    "Model": "모델",
    "Model Capability Radar": "모델 역량 레이더",
    "Model Health & Drift Detection": "모델 헬스 및 Drift 탐지",
    "Model Metrics Across Training Runs": "학습 실행별 모델 지표",
    "Model Metrics Across Training Runs (per run)": "학습 실행별 모델 지표 (실행 단위)",
    "Model Metrics Comparison": "모델 지표 비교",
    "Model Monitoring": "모델 모니터링",
    "Model Monitoring Dashboard": "모델 모니터링 대시보드",
    "Model Performance": "모델 성능",
    "Model Performance Comparison": "모델 성능 비교",
    "Model Performance Metrics Over Time": "시간에 따른 모델 성능 지표",
    "Model Performance Radar": "모델 성능 레이더",
    "Model Performance Radar (MLflow Runs)": "모델 성능 레이더 (MLflow 실행)",
    "Model Performance Summary": "모델 성능 요약",
    "Model Type": "모델 유형",
    "Model Type Usage in Recent Scoring": "최근 채점에서 모델 유형 사용",
    "Model training & artifact storage": "모델 학습 및 artifact 저장",
    "Monitoring Configuration": "모니터링 설정",
    "Monitoring Settings": "모니터링 설정",
    "Multiple Comparison Correction": "다중 비교 보정",
    "Navigation": "메뉴",
    "New Customers per Cohort": "코호트별 신규 고객 수",
    "No": "아니오",
    "No (showing cached runs)": "아니오 (캐시된 실행 표시 중)",
    "No CLV prediction data available.": "CLV 예측 데이터 없음.",
    "No Effect": "효과 없음",
    "No MLflow run data available.": "MLflow 실행 데이터 없음.",
    "No budget optimization data available.": "예산 최적화 데이터 없음.",
    "No cohort analysis data available.": "코호트 분석 데이터 없음.",
    "No drift detection history available yet.": "Drift 탐지 이력이 아직 없음.",
    "No drift detection history available.": "Drift 탐지 이력 없음.",
    "No experiment run data available.": "실험 실행 데이터 없음.",
    "No experiments logged yet - launch your first A/B test from the Retention Campaign Builder (Page 10) and re-run the pipeline to populate this view. The Power Analysis & Sample Size Calculator below is still usable for planning.": "아직 등록된 실험이 없습니다 — 유지 캠페인 빌더(페이지 10)에서 첫 A/B 테스트를 시작하고 파이프라인을 다시 실행하면 이 화면이 채워집니다. 아래 검정력 분석 & 샘플 크기 계산기는 계획용으로 계속 사용 가능합니다.",
    "No experiments returned by the MLflow server. Run history below is loaded from cached artifacts.": "MLflow 서버에서 반환된 실험이 없습니다. 아래 실행 이력은 캐시된 artifact에서 로드됩니다.",
    "No login": "로그인 없음",
    "No model performance metrics available.": "모델 성능 지표 없음.",
    "No offers match the selected filters.": "선택한 필터에 맞는 오퍼 없음.",
    "No purchase": "구매 없음",
    "No recommendation type data available.": "추천 유형 데이터 없음.",
    "No recommendations available.": "추천 없음.",
    "No retention offer data available.": "유지 오퍼 데이터 없음.",
    "No scoring throughput data available.": "채점 처리량 데이터 없음.",
    "No segmentation data available.": "세그멘테이션 데이터 없음.",
    "No treated actions available — uplift-by-action chart requires at least one non-`no_action` recommendation.": "처치 액션이 없음 — 액션별 업리프트 차트는 최소 1개 이상의 비-`no_action` 추천이 필요합니다.",
    "No uplift data available for recommendations.": "추천을 위한 업리프트 데이터 없음.",
    "No uplift data available.": "업리프트 데이터 없음.",
    "No uplift modeling data available.": "업리프트 모델링 데이터 없음.",
    "Non-healthy subsystems": "비정상 서브시스템",
    "Not Significant": "유의하지 않음",
    "Note: high_value_persuadable receives only": "참고: high_value_persuadable은 단지 다음만 받음:",
    "Number of Drifted Features": "Drift 발생 피처 수",
    "Number of Features": "피처 수",
    "Numbers shown are illustrative; they do NOT represent production performance.": "표시된 숫자는 예시용입니다; 운영 성능을 나타내지 않습니다.",
    "Of": "/",
    "Offer Type": "오퍼 유형",
    "Offer Type Distribution": "오퍼 유형 분포",
    "Offers Generated": "생성된 오퍼",
    "Only": "단지",
    "Operator": "연산자",
    "Overview": "개요",
    "PASSED": "통과",
    "PSI & KS Statistics Over Time": "시간에 따른 PSI 및 KS 통계",
    "PSI Drift Trend": "PSI Drift 추세",
    "PSI Trend (Population Stability Index)": "PSI 추세 (Population Stability Index)",
    "PSI Value": "PSI 값",
    "Page on-call and open an incident.": "On-call 호출 및 인시던트 발행.",
    "Peak Requests/min": "최고 요청/분",
    "Pending Messages": "대기 메시지",
    "Per-metric thresholds: no degradation flagged for": "지표별 임계값: 다음에 대해 저하 감지 안 됨:",
    "Performance Comparison": "성능 비교",
    "Performance Degradation Alerts": "성능 저하 알림",
    "Performance degradation detected for": "다음에서 성능 저하 감지:",
    "Performance degradation: drift threshold breached for": "성능 저하: 다음에 대해 drift 임계값 초과:",
    "Performance metrics for": "다음에 대한 성능 지표:",
    "Period": "기간",
    "Period-over-Period Retention Change": "기간별 유지율 변화",
    "Periods Tracked": "추적 기간 수",
    "Personalized Recommendations": "개인화된 추천",
    "Personalized Retention Offer Recommendations": "개인화된 유지 오퍼 추천",
    "Persuadable": "Persuadable",
    "Persuadable Customers": "Persuadable 고객",
    "Pipeline idle — no artifacts or models.": "파이프라인 유휴 — artifact 또는 모델 없음.",
    "Power": "검정력",
    "Power Analysis & Sample Size Calculator": "검정력 분석 및 샘플 크기 계산기",
    "Power Curve: Sample Size vs Statistical Power": "검정력 곡선: 샘플 크기 vs 통계적 검정력",
    "Power vs Sample Size": "검정력 vs 샘플 크기",
    "Precision": "정밀도",
    "Predicted": "예측",
    "Predicted CLV": "예측 CLV",
    "Predicted CLV (KRW)": "예측 CLV (KRW)",
    "Predicted Churn Rate": "예측 이탈률",
    "Predicted No": "예측 No",
    "Predicted Uplift (if treated)": "예측 업리프트 (처치 시)",
    "Predicted Uplift Distribution by Action Type": "액션 유형별 예측 업리프트 분포",
    "Predicted Yes": "예측 Yes",
    "Primary Model": "주요 모델",
    "Priority Score": "우선순위 점수",
    "Priority Score Distribution": "우선순위 점수 분포",
    "Priority Score vs Expected Uplift": "우선순위 점수 vs 예상 업리프트",
    "Priority vs treated reconciliation": "우선순위 vs 처치 일치 확인",
    "Priority-Ranked Retention Actions": "우선순위 정렬 유지 액션",
    "Proportion": "비율",
    "Proportion of Risk Levels within Each Segment": "세그먼트별 위험 등급 비율",
    "Quick Recommendation Lookup": "빠른 추천 조회",
    "ROC Curves": "ROC 곡선",
    "ROC Curves - Model Comparison": "ROC 곡선 - 모델 비교",
    "ROI (budget envelope)": "ROI (예산 envelope)",
    "ROI (treated only)": "ROI (처치 고객만)",
    "ROI (x)": "ROI (x)",
    "ROI Multiplier": "ROI 배수",
    "ROI Multiplier by Channel": "채널별 ROI 배수",
    "ROI by Offer Type": "오퍼 유형별 ROI",
    "ROI by Segment": "세그먼트별 ROI",
    "ROI by Segment (sorted)": "세그먼트별 ROI (정렬됨)",
    "Ratio": "비율",
    "Real confusion-matrix data missing — run `python -m src.main --mode all` to regenerate `results/confusion_matrices.json`. ": "실제 혼동 행렬 데이터 누락 — `python -m src.main --mode all`을 실행하여 `results/confusion_matrices.json`을 재생성하세요. ",
    "Real drift history missing — run the pipeline to populate `results/drift_history.csv` (or `monitoring_report.json`).": "실제 drift 이력 누락 — 파이프라인을 실행하여 `results/drift_history.csv`(또는 `monitoring_report.json`)를 채우세요.",
    "Real scoring throughput missing — run pipeline to populate `results/scoring_throughput.csv`.": "실제 채점 처리량 누락 — 파이프라인을 실행하여 `results/scoring_throughput.csv`를 채우세요.",
    "Real survival artifacts missing — run `python -m src.main --mode all` to generate `results/survival_data.csv` and `results/survival_curves.json`. The previous Kaplan-Meier / hazard / median-duration KPIs were derived from churn predictions (duration = 365 × (1 − churn_prob)) and are not a fitted Cox PH output.": "실제 생존 분석 artifact 누락 — `python -m src.main --mode all`을 실행하여 `results/survival_data.csv`와 `results/survival_curves.json`을 생성하세요. 이전의 Kaplan-Meier / hazard / 중앙 기간 KPI는 이탈 예측에서 도출된(duration = 365 × (1 − churn_prob)) 값이며, 학습된 Cox PH 출력이 아닙니다.",
    "Real survival-curve data missing — run `python -m src.main --mode all` to generate `results/survival_curves.json`.": "실제 생존 곡선 데이터 누락 — `python -m src.main --mode all`을 실행하여 `results/survival_curves.json`을 생성하세요.",
    "Real-Time Scoring": "실시간 채점",
    "Real-Time Scoring & Recommendations": "실시간 채점 및 추천",
    "Real-time health monitoring for all system components: streaming pipeline, ML tracking, and model serving.": "모든 시스템 구성요소(스트리밍 파이프라인, ML 추적, 모델 서빙)의 실시간 헬스 모니터링.",
    "Reason": "사유",
    "Recall": "재현율",
    "Recent Scoring History": "최근 채점 이력",
    "Recommendation": "추천",
    "Recommendation Counts by Type": "유형별 추천 수",
    "Recommendation Details": "추천 세부사항",
    "Recommendation Distribution": "추천 분포",
    "Recommendation Engine Configuration": "추천 엔진 설정",
    "Recommendation Type Distribution": "추천 유형 분포",
    "Recommendation Types by Segment": "세그먼트별 추천 유형",
    "Recommendations": "추천",
    "Recommendations by Type": "유형별 추천",
    "Recommended Action": "추천 액션",
    "Recommended Actions by Customer Segment": "고객 세그먼트별 추천 액션",
    "Red Alerts": "Red 알림",
    "Redis Streaming": "Redis 스트리밍",
    "Redis not connected. Stream metrics unavailable. Start Redis with `docker-compose up redis`.": "Redis에 연결되지 않음. Stream 지표 사용 불가. `docker-compose up redis`로 Redis를 시작하세요.",
    "Redis: Connected": "Redis: 연결됨",
    "Redis: Unavailable": "Redis: 사용 불가",
    "Refresh Data": "데이터 새로고침",
    "Registered Experiments": "등록된 실험",
    "Relative Lift by Experiment": "실험별 상대 lift",
    "Request queue depth (current)": "요청 큐 깊이 (현재)",
    "Requests per Minute": "분당 요청",
    "Requests/min": "요청/분",
    "Required Sample Size (per group)": "그룹당 필요 샘플 크기",
    "Response Classification by Segment": "세그먼트별 반응 분류",
    "Response Latency & Error Rate": "응답 지연시간 및 오류율",
    "Response queue depth (current)": "응답 큐 깊이 (현재)",
    "Retained Customers": "유지된 고객",
    "Retention Actions by Segment": "세그먼트별 유지 액션",
    "Retention Change Between Periods (Average)": "기간 간 유지율 변화 (평균)",
    "Retention Curves by Cohort": "코호트별 유지 곡선",
    "Retention Heatmap": "유지율 히트맵",
    "Retention Matrix (Raw Data)": "유지율 행렬 (원본 데이터)",
    "Retention Rate (%)": "유지율 (%)",
    "Retention Rate Over Time by Cohort": "시간에 따른 코호트별 유지율",
    "Retention offer breakdown not yet computed — top KPIs above show full population stats from real recommendations.csv.": "유지 오퍼 분해가 아직 계산되지 않음 — 위의 상위 KPI는 실제 recommendations.csv의 전체 모집단 통계를 표시.",
    "Revenue Saved": "절감된 매출",
    "Revenue Saved Waterfall by Segment": "세그먼트별 절감 매출 waterfall",
    "Right-censoring artifact possible above ~85% ratio — the displayed median is bounded by the observation window.": "~85% 비율 초과 시 우측 절단 artifact 가능 — 표시된 중앙값은 관측 윈도우에 의해 제한됩니다.",
    "Right-censoring artifact possible above ~85% ratio.": "~85% 비율 초과 시 우측 절단 artifact 가능.",
    "Risk Level": "위험 등급",
    "Risk Level Breakdown": "위험 등급 분해",
    "Risk Level Distribution": "위험 등급 분포",
    "Risk Level Distribution by Segment": "세그먼트별 위험 등급 분포",
    "Risk Level Distribution within Segments": "세그먼트 내 위험 등급 분포",
    "Risk Score": "위험 점수",
    "Run": "실행",
    "Run Details": "실행 세부사항",
    "SLO BREACH — error rate": "SLO 위반 — 오류율",
    "SaaS SLO target.": "SaaS SLO 목표.",
    "Sample Size per Group": "그룹당 샘플 크기",
    "Scenario Comparison: Allocation vs ROI": "시나리오 비교: 배정 vs ROI",
    "Score": "점수",
    "Scoring Error Rate": "채점 오류율",
    "Scoring Quality Metrics": "채점 품질 지표",
    "Scoring Requests per Minute": "분당 채점 요청",
    "Scoring Throughput": "채점 처리량",
    "Scoring Throughput & Latency": "채점 처리량 및 지연시간",
    "Scoring Throughput (last 24h)": "채점 처리량 (지난 24h)",
    "Scoring Volume (last 24h, demo)": "채점 볼륨 (지난 24h, 데모)",
    "Scoring Volume Over Time": "시간에 따른 채점 볼륨",
    "Segment": "세그먼트",
    "Segment CLV vs Churn Risk (size = customers)": "세그먼트 CLV vs 이탈 위험 (크기 = 고객 수)",
    "Segment Churn Risk Analysis": "세그먼트 이탈 위험 분석",
    "Segment Definitions & Retention Actions": "세그먼트 정의 및 유지 액션",
    "Segment Distribution": "세그먼트 분포",
    "Segment Statistics": "세그먼트 통계",
    "Segment Summary": "세그먼트 요약",
    "Segment or recommendation type data not available.": "세그먼트 또는 추천 유형 데이터 없음.",
    "Segment x Risk Level Cross-Tabulation": "세그먼트 × 위험 등급 교차표",
    "Segment-Level Breakdown": "세그먼트별 분해",
    "Segment-Level Churn Risk Analysis": "세그먼트별 이탈 위험 분석",
    "Select Customer ID": "고객 ID 선택",
    "Select Page": "페이지 선택",
    "Service Health": "서비스 헬스",
    "Showing": "표시 중",
    "Showing latest snapshot only — a trend chart is suppressed to avoid a degenerate single-point line.": "최신 스냅샷만 표시 — 단일 점 추세선이 되는 것을 방지하기 위해 추세 차트를 숨겼습니다.",
    "Significance Level (α)": "유의수준 (α)",
    "Significant Results": "통계적 유의 결과",
    "Sleeping Dog": "Sleeping Dog",
    "Sleeping Dogs": "Sleeping Dogs",
    "Smallest effect size you want to detect": "감지하고자 하는 최소 효과 크기",
    "Statistical Details": "통계 세부사항",
    "Statistical Power": "통계적 검정력",
    "Statistical Power vs p-value": "통계적 검정력 vs p-value",
    "Statistically Significant": "통계적으로 유의",
    "Stream": "Stream",
    "Stream Lengths": "Stream 길이",
    "Stream Metrics": "Stream 지표",
    "Streaming Pipeline Status": "스트리밍 파이프라인 상태",
    "Sum of predicted Customer Lifetime Value across all customers. Compact display (B/M/K) avoids overflow truncation in the KPI tile.": "모든 고객의 예측 생애 가치 합계. 컴팩트 표시(B/M/K)로 KPI 타일의 overflow 절단을 방지합니다.",
    "Sure Thing": "Sure Thing",
    "Survival Analysis": "생존 분석",
    "Survival Model Configuration": "생존 모델 설정",
    "Survival Probability": "생존 확률",
    "Synthetic data": "합성 데이터",
    "System Configuration": "시스템 설정",
    "System Health": "시스템 헬스",
    "System Issues Detected": "시스템 문제 감지",
    "System Overview & Health": "시스템 개요 및 헬스",
    "System Status": "시스템 상태",
    "Target": "목표",
    "Target Power (1-β)": "목표 검정력 (1-β)",
    "This is the budget-envelope ROI; see the caption below for the mean of per-segment ROIs.": "이는 예산 envelope ROI입니다; 세그먼트별 ROI 평균은 아래 캡션 참조.",
    "Threshold": "임계값",
    "Throughput Summary": "처리량 요약",
    "Throughput telemetry is stale": "처리량 텔레메트리가 stale",
    "Time": "시간",
    "Timestamp": "타임스탬프",
    "Top": "상위",
    "Top 10 Churn Prediction Features": "이탈 예측 피처 상위 10개",
    "Top 10 Customers by CLV": "CLV 기준 상위 10명 고객",
    "Top 10 Feature Importance Scores": "피처 중요도 점수 상위 10개",
    "Top 10 Persuadable Customers": "Persuadable 고객 상위 10명",
    "Top Action Type": "최다 액션 유형",
    "Top Priority Customers for Retention": "유지 우선순위 상위 고객",
    "Top Priority Recommendations": "우선순위 상위 추천",
    "Total": "총",
    "Total Allocated": "총 배정액",
    "Total Budget (KRW)": "총 예산 (KRW)",
    "Total CLV": "총 CLV",
    "Total CLV by Segment (n>=5 only)": "세그먼트별 총 CLV (n>=5만)",
    "Total Campaign Cost": "총 캠페인 비용",
    "Total Checks": "총 점검 수",
    "Total Cohorts": "총 코호트 수",
    "Total Cost": "총 비용",
    "Total Cost by Offer Type": "오퍼 유형별 총 비용",
    "Total Customers": "총 고객 수",
    "Total Drift Checks": "총 Drift 점검 수",
    "Total Estimated Cost": "총 추정 비용",
    "Total Experiments": "총 실험 수",
    "Total Offers": "총 오퍼 수",
    "Total Participants Needed": "필요 총 참가자 수",
    "Total Recommendations": "총 추천 수",
    "Total Runs": "총 실행 수",
    "Total Scores (lifetime)": "총 점수 (lifetime)",
    "Total Segments": "총 세그먼트 수",
    "Total Train Time": "총 학습 시간",
    "Total Training Time": "총 학습 시간",
    "Training Efficiency": "학습 효율성",
    "Training Run History": "학습 실행 이력",
    "Training Time (s)": "학습 시간 (s)",
    "Training Time (seconds)": "학습 시간 (초)",
    "Training Time by Model Type": "모델 유형별 학습 시간",
    "Treatable Customers": "처치 가능 고객",
    "Treatment": "처치군",
    "Treatment Churn": "처치군 이탈",
    "True Positive Rate": "참양성률 (TPR)",
    "True median may be longer.": "실제 중앙값은 더 길 수 있습니다.",
    "UNKNOWN": "알 수 없음",
    "Unknown": "알 수 없음",
    "Unobserved cells (cohorts whose follow-up window has not yet elapsed) are rendered as \"—\" rather than zero-filled — closes iter9 P04 #17.": "관측되지 않은 셀(추적 기간이 경과하지 않은 코호트)은 0으로 채우지 않고 \"—\"로 표시됩니다.",
    "Uplift & Treatment Effect by Segment": "세그먼트별 업리프트 및 처치 효과",
    "Uplift Modeling": "업리프트 모델링",
    "Uplift Modeling Results": "업리프트 모델링 결과",
    "Uplift Multiplier": "업리프트 배수",
    "Uplift Score Distribution": "업리프트 점수 분포",
    "Uplift Score vs Treatment Effect": "업리프트 점수 vs 처치 효과",
    "Uplift Score vs Treatment Effect by Segment": "세그먼트별 업리프트 점수 vs 처치 효과",
    "Uplift by Segment": "세그먼트별 업리프트",
    "What-If Scenario Comparison": "What-If 시나리오 비교",
    "When running multiple experiments simultaneously, p-values should be corrected to control the family-wise error rate.": "여러 실험을 동시에 진행할 때는 family-wise error rate 통제를 위해 p-value 보정이 필요합니다.",
    "Yellow Alerts": "Yellow 알림",
    "Yellow Warnings": "Yellow 경고",
    "Yes": "예",
    "Yes (idle — no runs logged)": "예 (유휴 — 기록된 실행 없음)",
    "Yes (idle — no traffic)": "예 (유휴 — 트래픽 없음)",
    "`no_action` excluded: realized uplift on untreated customers is 0 by definition; including its 'predicted uplift if treated' on the same axis as treated actions is misleading.": "`no_action` 제외: 미처치 고객의 실현된 업리프트는 정의상 0이며, '처치 시 예측 업리프트'를 처치 액션과 같은 축에 표시하면 오해를 부릅니다.",
    "approaching thresholds": "임계값에 접근",
    "are approaching thresholds.": "임계값에 접근 중.",
    "artifact not found": "artifact를 찾을 수 없음",
    "customers above threshold": "임계값 초과 고객",
    "day(s) old.": "일 경과.",
    "days": "일",
    "despite a ~": "그러나 ~",
    "for any offer in the catalog. Total treated across all priorities:": "카탈로그의 어떤 오퍼에 대해서도. 모든 우선순위에서의 총 처치 수:",
    "get `no_action` because their": "는 다음 이유로 `no_action`을 받음:",
    "high-priority customers,": "고우선순위 고객,",
    "is": "는",
    "is within the": "는 다음 범위 내:",
    "last sample": "마지막 샘플",
    "mode": "모드",
    "monthly cohorts are available. Production cohort analysis typically uses ≥6–12 cohorts; generate more historical data for trend reliability.": "월별 코호트만 사용 가능합니다. 운영 환경 코호트 분석은 일반적으로 6-12개 이상의 코호트를 사용합니다; 추세 신뢰도를 위해 더 많은 과거 데이터를 생성하세요.",
    "not found": "찾을 수 없음",
    "observation horizon": "관측 시계",
    "of total CLV": "총 CLV 중",
    "p-value": "p-value",
    "predicted uplift × CLV did not exceed the cost threshold": "예측 업리프트 × CLV가 비용 임계값을 초과하지 않음",
    "ratio": "비율",
    "receive a treatment offer;": "처치 오퍼 수신;",
    "recommendations": "추천",
    "replace with live telemetry before production. Latest sample is": "운영 전 실제 텔레메트리로 교체하세요. 최신 샘플은",
    "right-censored at observation window": "관측 윈도우에서 우측 절단됨",
    "rows": "행",
    "see ROI by Segment chart. The headline above uses the aggregate revenue_saved / total_allocated, which is the production-relevant scope (iter11 fix for verify_v2 #5).": "세그먼트별 ROI 차트 참조. 상단 헤드라인은 집계 revenue_saved / total_allocated를 사용하며, 이는 운영 환경에서 의미 있는 범위입니다.",
    "subsystems healthy": "서브시스템 정상",
    "the SaaS SLO target of": "SaaS SLO 목표인",
    "threshold": "임계값",
    "training run(s) available — showing an index-based comparison rather than a temporal trend.": "개의 학습 실행 사용 가능 — 시간적 추세 대신 인덱스 기반 비교를 표시합니다.",
    "unknown": "알 수 없음",
    "x ROI.": "배 ROI.",
    "ℹ️ AUC spread across the three models is {auc_margin:.4f} (<0.005). No DeLong significance test was run; the \"Best Model\" label is indicative only.": "ℹ️ 세 모델의 AUC 격차는 {auc_margin:.4f} (<0.005)입니다. DeLong 유의성 검정은 실행되지 않았으며; \"최고 모델\" 라벨은 참고용입니다.",
    "⚠ Event Rate per segment is derived from current outcome labels — these uplift segments (sure_thing / lost_cause / persuadable / sleeping_dog) are defined post-hoc using churn outcome, so the binary 0% / 100% pattern is tautological and NOT a model finding. Use **Avg Survival Probability** (Cox PH-derived) above for proper per-segment risk.": "⚠ 세그먼트별 이벤트 비율은 현재 결과 라벨에서 도출됨 — 이러한 업리프트 세그먼트(sure_thing / lost_cause / persuadable / sleeping_dog)는 이탈 결과를 사용해 사후 정의되므로, 0% / 100% 이항 패턴은 동의어 반복이며 모델 발견이 아닙니다. 적절한 세그먼트별 위험은 위의 **Avg Survival Probability**(Cox PH 도출)를 사용하세요.",
    "⚠️ Retention monotonicity violations detected — retention must be non-increasing within a cohort by construction. Affected cells are flagged with red asterisks in the heatmap below: ": "⚠️ 유지율 단조성 위반 감지 — 코호트 내 유지율은 정의상 비증가여야 합니다. 영향받는 셀은 아래 히트맵에서 빨간 별표로 표시: ",
}


def tr(key: str, lang: str = "en") -> str:
    """Translate a UI label.

    Args:
        key: English source string.
        lang: "en" (passthrough) or "ko" (lookup in I18N_KO, fallback to key).
    Returns:
        Translated string, or the key itself when no translation exists.
    """
    if lang == "ko":
        return I18N_KO.get(key, key)
    return key


def get_lang() -> str:
    """Return the currently selected dashboard language.

    Reads from Streamlit session state if available, defaults to 'en'.
    """
    try:
        import streamlit as st  # local import so non-Streamlit callers don't break
        return st.session_state.get("lang", "en")
    except Exception:
        return "en"

PAGE_ICONS = {
    "Demo": "\U0001f680",                  # rocket
    "Overview": "\U0001f4ca",              # bar chart
    "Churn Analytics": "\U0001f50d",       # magnifying glass
    "Model Performance": "\U0001f3af",     # target
    "Customer Segmentation": "\U0001f465", # people
    "Cohort Analysis": "\U0001f4c5",       # calendar
    "Budget Optimization": "\U0001f4b0",   # money bag
    "A/B Testing": "\U0001f9ea",           # test tube
    "Survival Analysis": "\U0001f4c8",     # chart increasing
    "Model Monitoring": "\U0001f6e1",      # shield
    "Recommendations": "\U0001f4e9",       # envelope
    "CLV Prediction": "\U0001f4b5",        # dollar bill
    "Uplift Modeling": "\U0001f4c8",       # chart increasing
    "CLV & Retention Campaign": "\U0001f3af",  # target
    "Real-Time Scoring": "\u26a1",         # lightning
    "MLflow Experiments": "\U0001f52c",    # microscope
    "System Health": "\U0001f3e5",          # hospital
}

# Default color palette
DEFAULT_PALETTE = [
    "#2ecc71", "#3498db", "#9b59b6", "#e67e22",
    "#e74c3c", "#1abc9c", "#f39c12", "#2c3e50",
    "#16a085", "#8e44ad",
]

RISK_COLORS = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e67e22",
    "critical": "#e74c3c",
}

# Required columns for churn prediction DataFrame
REQUIRED_PREDICTION_COLUMNS = [
    "customer_id",
    "churn_probability",
    "risk_level",
    "segment",
]

APP_TITLE = "Churn Prediction & Retention Dashboard"


# =========================================================================
# Formatting helpers
# =========================================================================

def _is_missing(x: Any) -> bool:
    """Return True if x is None, NaN, or +/-infinity."""
    if x is None:
        return True
    if isinstance(x, float):
        if x != x:  # NaN
            return True
        if x in (float("inf"), float("-inf")):
            return True
    return False


def format_currency(value: float, currency: str = "KRW") -> str:
    """Format a numeric value as currency with commas.

    Args:
        value: Numeric value to format.
        currency: Currency code (default: KRW).

    Returns:
        Formatted currency string, e.g. '50,000,000 KRW'.
    """
    if _is_missing(value):
        return "—"
    return f"{value:,.0f} {currency}"


def format_currency_krw(x: Any) -> str:
    """Format a KRW amount with B/M/K suffix and won symbol.

    Closes the ``57,936,514,970 ...`` ellipsis-truncation issue audited on
    Page 10 by collapsing very large headline numbers to a compact, human-
    readable form (e.g. ``₩57.94B``).

    Args:
        x: Numeric KRW amount. ``None``/``NaN``/``inf`` render as ``"—"``.

    Returns:
        Compact currency string with won symbol and scale suffix.
    """
    if _is_missing(x):
        return "—"
    n = float(x)
    if abs(n) >= 1_000_000_000:
        return f"₩{n / 1_000_000_000:,.2f}B"
    if abs(n) >= 1_000_000:
        return f"₩{n / 1_000_000:,.1f}M"
    if abs(n) >= 1_000:
        return f"₩{n / 1_000:,.1f}K"
    return f"₩{n:,.0f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g. 0.1234).
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string, e.g. '12.34%'.
    """
    if _is_missing(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def format_count(
    value: Any,
    integer: bool = True,
    suffix: str = "",
) -> str:
    """Format a count for KPI cards.

    Closes the ``Customers Retained = 122.29548658078494`` 14-decimal float
    leak audited on Page 12: when ``integer=True`` (default) the value is
    floored to an int and rendered with thousand separators, regardless of
    the raw float precision flowing in from the model layer.

    Args:
        value: Numeric value. ``None``/``NaN``/``inf`` render as ``"—"``.
        integer: If True, floor to int and render with comma grouping. If
            False, render as a 1-decimal fixed-point number.
        suffix: Optional unit suffix appended to the formatted output
            (e.g. ``" customers"``).

    Returns:
        Formatted count string. Backward-compatible with the legacy
        single-argument call ``format_count(1234)`` returning ``"1,234"``.
    """
    if _is_missing(value):
        return "—"
    if integer:
        out = f"{int(value):,}"
    else:
        out = f"{float(value):,.1f}"
    return f"{out}{suffix}" if suffix else out


def compute_overall_roi(
    revenue_saved: float,
    cost_or_budget: float,
    scope_label: str = "budget",
) -> Dict[str, Any]:
    """Single source of truth for the "Overall ROI" KPI.

    Closes the iter9 audit finding that Pages 05 / 09 / 12 each report a
    different "Overall ROI" (3.5x / 9.0x / 3.8x) for the same campaign
    because each page silently uses a different denominator. This helper
    forces the caller to declare the scope, returns the pre-formatted
    display string, and bundles a tooltip showing the actual division so
    the dashboard can footnote every ROI tile.

    Args:
        revenue_saved: Numerator — expected revenue retained (KRW).
        cost_or_budget: Denominator — interpretation depends on
            ``scope_label``. ``0`` or ``None`` short-circuits to "—".
        scope_label: One of:

            * ``"budget"``     — denominator is the full LP budget envelope.
            * ``"treated"``    — denominator is the cost actually issued
              (sum of issued offers).
            * ``"segment_avg"``— caller has pre-computed the mean of
              per-segment ROIs and is passing it as ``revenue_saved`` /
              ``cost_or_budget = 1`` (or supplies the mean directly via
              ``revenue_saved`` and ``cost_or_budget=1``).

    Returns:
        Dict with keys ``value`` (float or None), ``display`` (str such as
        ``"3.50x"`` or ``"—"``), ``label`` (human-readable scope), and
        ``tooltip`` (the literal division shown to operators).
    """
    label_map = {
        "budget": "ROI (budget envelope)",
        "treated": "ROI (treated only)",
        "segment_avg": "Avg per-segment ROI",
    }
    label = label_map.get(scope_label, "ROI")

    if cost_or_budget is None or cost_or_budget == 0 or _is_missing(cost_or_budget):
        return {
            "value": None,
            "display": "—",
            "label": label,
            "tooltip": "denominator is zero",
        }
    if _is_missing(revenue_saved):
        return {
            "value": None,
            "display": "—",
            "label": label,
            "tooltip": "numerator is missing",
        }

    val = float(revenue_saved) / float(cost_or_budget)
    return {
        "value": val,
        "display": f"{val:.2f}x",
        "label": label,
        "tooltip": f"{float(revenue_saved):,.0f} ÷ {float(cost_or_budget):,.0f}",
    }


def drift_trend_guard(
    timeseries: Any,
    min_points: int = 5,
) -> Tuple[bool, str]:
    """Validate that a time series is dense enough to plot as a trend.

    Closes the iter9 audit finding that Pages 08 / 13c / 14 render
    "trend over time" charts whose x-axis spans ~1.5 ms (a single
    observation drawn as a line). Callers should branch on the returned
    flag and surface ``st.info(message)`` instead of a misleading line
    chart when the series is too sparse.

    Args:
        timeseries: Iterable of timestamps or a pandas Series of timestamps
            for which a trend chart is being considered. ``None`` is
            treated as length 0.
        min_points: Minimum observation count required before a "trend"
            framing is allowed (default 5).

    Returns:
        Tuple ``(ok, message)``. ``ok=True`` and an empty message indicate
        the caller may render a normal trend chart. ``ok=False`` returns a
        human-readable explanation suitable for a Streamlit info banner.
    """
    if timeseries is None:
        return False, (
            f"Insufficient history — need ≥{min_points} observations, have 0."
        )
    try:
        n = len(timeseries)
    except TypeError:
        try:
            n = sum(1 for _ in timeseries)
        except Exception:
            n = 0
    if n < min_points:
        return False, (
            f"Insufficient history — need ≥{min_points} observations, have {n}."
        )

    # If the input is genuinely timestamp-like, also reject sub-hour spans
    # which the iter9 audit found masquerading as trends (~0.1 ms / ~1.5 ms
    # windows). We only apply this check when the input already looks like
    # a datetime sequence to avoid pandas silently treating raw integers as
    # nanosecond epochs.
    looks_like_timestamps = False
    try:
        if isinstance(timeseries, pd.Series):
            looks_like_timestamps = pd.api.types.is_datetime64_any_dtype(
                timeseries.dtype
            )
        elif isinstance(timeseries, pd.DatetimeIndex):
            looks_like_timestamps = True
        else:
            sample = next(iter(timeseries), None)
            if isinstance(sample, (pd.Timestamp, dt_module.datetime, dt_module.date)):
                looks_like_timestamps = True
            elif isinstance(sample, str):
                # Try a strict parse on the first element; if it fails the
                # sequence is not treated as timestamps.
                parsed = pd.to_datetime(sample, errors="coerce")
                looks_like_timestamps = parsed is not pd.NaT and not pd.isna(parsed)
    except Exception:
        looks_like_timestamps = False

    if looks_like_timestamps:
        try:
            ts = pd.to_datetime(pd.Series(list(timeseries)), errors="coerce")
            ts = ts.dropna()
            if len(ts) >= 2:
                span = ts.max() - ts.min()
                seconds = span.total_seconds()
                # iter14 fix: all-same-timestamp detection. ``run_monitor``
                # writes every drift check in a single batch, so a 34-row
                # history can still have all timestamps within microseconds
                # of each other and produce a degenerate vertical line.
                if seconds < 5:
                    return False, (
                        "All drift checks come from one pipeline invocation "
                        "(timestamps span <5s) — run `python -m src.main "
                        "--mode monitor` multiple times to build trend "
                        "history."
                    )
                if seconds < 3600:
                    return False, (
                        f"Trend window is {seconds:.1f}s — too short to be a "
                        f"real trend (require ≥1 hour)."
                    )
        except Exception:
            # Non-timestamp sequence — point-count check above is sufficient.
            pass

    return True, ""


# =========================================================================
# Risk classification
# =========================================================================

def classify_risk(
    probability: float,
    thresholds: Tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> str:
    """Classify churn probability into risk level.

    Args:
        probability: Churn probability between 0 and 1.
        thresholds: Tuple of (low_max, medium_max, high_max) boundaries.
            Values <= thresholds[0] are 'low',
            <= thresholds[1] are 'medium',
            <= thresholds[2] are 'high',
            above are 'critical'.

    Returns:
        Risk level string: 'low', 'medium', 'high', or 'critical'.
    """
    low_max, med_max, high_max = thresholds
    if probability <= low_max:
        return "low"
    elif probability <= med_max:
        return "medium"
    elif probability <= high_max:
        return "high"
    else:
        return "critical"


def get_risk_color(risk_level: str) -> str:
    """Get the color code for a risk level.

    Args:
        risk_level: One of 'low', 'medium', 'high', 'critical'.

    Returns:
        Hex color string.
    """
    return RISK_COLORS.get(risk_level, "#95a5a6")


# =========================================================================
# Data validation
# =========================================================================

def validate_predictions(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate a churn predictions DataFrame.

    Checks for required columns and non-empty data.

    Args:
        df: Predictions DataFrame to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if df is None or df.empty:
        errors.append("Predictions DataFrame is empty or None.")
        return False, errors

    missing = set(REQUIRED_PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    return len(errors) == 0, errors


def safe_get_column(
    df: pd.DataFrame,
    column: str,
    default: Any = 0,
) -> pd.Series:
    """Safely get a column from a DataFrame with a default.

    Args:
        df: Source DataFrame.
        column: Column name to retrieve.
        default: Default value if column is missing.

    Returns:
        Series with column values or filled with default.
    """
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


# =========================================================================
# Chart helpers
# =========================================================================

def get_color_palette() -> List[str]:
    """Get the default color palette for charts.

    Returns:
        List of hex color strings.
    """
    return list(DEFAULT_PALETTE)


def get_segment_colors(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract segment-to-color mapping from configuration.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict mapping segment name to hex color.
    """
    segments = config.get("segmentation", {}).get("segments", [])
    colors = {}
    for i, seg in enumerate(segments):
        name = seg.get("name", f"segment_{i}")
        color = seg.get("color", DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
        colors[name] = color
    return colors


def compute_kpi_delta(
    current: float,
    previous: float,
) -> float:
    """Compute percentage delta between current and previous KPI values.

    Args:
        current: Current period value.
        previous: Previous period value.

    Returns:
        Percentage change (e.g. 25.0 means +25%).
    """
    if previous == 0:
        return 0.0
    return round((current - previous) / previous * 100, 1)


# =========================================================================
# Page routing
# =========================================================================

def get_page_list() -> List[str]:
    """Get the ordered list of dashboard pages.

    Returns:
        List of page name strings (11 pages).
    """
    return list(PAGES)


def get_page_icon(page_name: str) -> str:
    """Get the icon for a dashboard page.

    Args:
        page_name: Name of the page.

    Returns:
        Emoji/icon string for the page.
    """
    return PAGE_ICONS.get(page_name, "\U0001f4cb")  # default: clipboard


# =========================================================================
# Config extraction helpers
# =========================================================================

def get_churn_definition(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract churn definition parameters from config.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict with no_purchase_days, no_login_days, operator.
    """
    churn_def = config.get("churn_definition", {})
    return {
        "no_purchase_days": churn_def.get("no_purchase_days", 30),
        "no_login_days": churn_def.get("no_login_days", 60),
        "operator": churn_def.get("operator", "OR"),
    }


def get_ensemble_weights(
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """Extract ensemble model weights from config.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Tuple of (ml_weight, dl_weight).
    """
    pipeline = config.get("pipeline", {})
    ml_w = pipeline.get("ensemble_weight_ml", 0.6)
    dl_w = pipeline.get("ensemble_weight_dl", 0.4)
    return ml_w, dl_w


def get_budget_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract budget configuration.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict with total_krw, currency keys.
    """
    budget = config.get("budget", {})
    return {
        "total_krw": budget.get("total_krw", 50_000_000),
        "currency": budget.get("currency", "KRW"),
    }


# =========================================================================
# Sidebar helpers
# =========================================================================

def build_sidebar_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build sidebar information dictionary from config.

    Collects churn definition, budget, and ensemble weight info
    for display in the Streamlit sidebar.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict with churn_definition, budget, ensemble_weights keys.
    """
    return {
        "churn_definition": get_churn_definition(config),
        "budget": get_budget_config(config),
        "ensemble_weights": {
            "ml": get_ensemble_weights(config)[0],
            "dl": get_ensemble_weights(config)[1],
        },
    }


def get_app_title() -> str:
    """Get the dashboard application title.

    Returns:
        Application title string.
    """
    return APP_TITLE


# =========================================================================
# Real-only artifact enforcement helpers (iter13)
# =========================================================================
#
# These helpers back the iter13 "real-only data" enforcement rollout. View
# code wraps each chart in ``assert_real_or_error`` to short-circuit
# rendering when ``data_loader`` reports the artifact is a fixture / fallback
# rather than a real materialised file. ``freshness_caption`` renders the
# accompanying "Last refresh: <ts>" footer using artifact metadata so the
# operator can tell which run produced the numbers.


def assert_real_or_error(st, artifact, label: str) -> bool:
    """Guard a Streamlit view against fixture/fallback artifacts.

    Helper for view code. If ``artifact.is_real`` is False, render
    ``st.error`` with a useful message and return False. Otherwise return
    True (caller proceeds to render the chart).

    Args:
        st: The Streamlit module (passed in to keep this helper import-free
            for non-dashboard callers and to ease unit-testing).
        artifact: An artifact object exposed by ``data_loader`` when called
            with ``as_artifact=True``. Expected to expose ``is_real``,
            ``reason``, and ``source_path`` attributes; missing attributes
            are tolerated for backwards compatibility.
        label: Human-readable label for the artifact (used in the error
            message), e.g. ``"Retention offers"``.

    Returns:
        True when the artifact is real (or backwards-compatibly assumed
        real because ``is_real`` is absent). False when a Streamlit error
        was rendered and the caller should bail out of the view block.

    Usage:
        offers = data_loader.load_retention_offers(as_artifact=True)
        if not assert_real_or_error(st, offers, "Retention offers"):
            return
        # ... render normally
    """
    if artifact is None:
        st.error(
            f"{label}: artifact loader returned None — check data_loader wiring."
        )
        return False
    is_real = getattr(artifact, "is_real", True)  # backwards-compat
    if not is_real:
        reason = getattr(artifact, "reason", "unknown")
        source = getattr(artifact, "source_path", "unknown")
        st.error(
            f"**{label} unavailable** — real artifact missing.\n\n"
            f"Source: `{source}`\n\n"
            f"Reason: {reason}\n\n"
            f"Run `python -m src.main --mode all` to regenerate."
        )
        return False
    return True


def freshness_caption(artifact, default_label: str = "Last refresh") -> str:
    """Render a "Last refresh: <ts>" caption from artifact metadata.

    Args:
        artifact: An artifact object (see ``assert_real_or_error``). May
            expose ``computed_at`` or ``mtime``. ``None`` returns an empty
            string so callers can unconditionally ``st.caption(...)`` the
            result without guarding.
        default_label: Caption prefix (default ``"Last refresh"``).

    Returns:
        A Markdown italic caption string, or empty string when the artifact
        is ``None``. When the artifact exists but exposes no timestamp the
        caption renders ``"unknown"`` so the gap is visible to operators.
    """
    if artifact is None:
        return ""
    ts = getattr(artifact, "computed_at", None) or getattr(artifact, "mtime", None)
    if ts is None:
        return f"_{default_label}: unknown_"
    return f"_{default_label}: {ts}_"
