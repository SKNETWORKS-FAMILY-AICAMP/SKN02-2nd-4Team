# Telecom Company’s                                                           Churn Prediction

<aside>
🙌🏻 **SK네트웍스 AI Family캠프 2기 4조**
구선아 김서연 박주희 이동건 진정현

</aside>

**Table of contents**

<aside>
<img src="https://www.notion.so/icons/command-line_purple.svg" alt="https://www.notion.so/icons/command-line_purple.svg" width="40px" /> **기술스택**

![Untitled](img/Untitled.png)

![Untitled](img/Untitled%201.png)

![Untitled](img/Untitled%202.png)

![Untitled](img/Untitled%203.png)

![Untitled](img/Untitled%204.png)

</aside>

# 📍 주제

---

## ▶ 주제 및 주제 선정 배경

    **주제
    탈퇴 고객 분석 및 예측을 활용한 고객 이탈률 개선 방안 제안**

    부주제
    탈퇴 고객 분석 모델링을 위한 수집 데이터 개선 방안

<aside>
💡 탈퇴 고객 예측 프로젝트를 통해 실무 감각을 키워보고자 실제 기업에서 데이터 분석을 활용하는 방식에 가까운 **영업/마케팅**과 **데이터 보수 및 관리**, 두 가지 시각으로 분석 진행

</aside>

## ▶ 활용 데이터셋

[Iranian Churn from UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

# 🗨️ 추측 및 탐색

---

## ▶ 상관관계 추측

추측 1. 장기 고객은 이탈률이 낮을 것이다.

추측 2. 연령대가 이탈률에 영향을 미칠 것이다.

추측 3. 고객 가치 지표가 높으면 이탈률이 낮을 것이다.

<aside>
🤔 정말 우리가 추측한 내용대로 가입 기간, 연령, 고객 가치가 이탈률에 영향을 미칠까?

</aside>

## ▶ 데이터 탐색

```python
data = pd.read_csv('data/Iranian_Churn.csv')
data.head()
```

![Untitled](img/Untitled%205.png)

📑 Call Failure
📑 Complains
📑 Subscription Length
📑 Charge Amount
📑 Seconds of Use
📑 Frequency of Use
📑 Frequency of SMS
📑 Distinct Called Numbers
📑 Age Group
📑 Tariff Plan
📑 Status
📑 Age
📑 Customer Value
📑 Churn

통화 실패 횟수
불만 여부 (0: 불만 없음 / 1: 불만 있음)
가입 기간 (개월)
청구 금액 (낮은 금액 0 - 9 높은 금액)
총 통화 시간 (초)
총 통화 횟수
총 SMS 발송 횟수
총 통화 상대 수
연령대 (1: 10대 - 5: 50대)
요금제 (1: 선불 / 2: 계약제)
상태 (1: 활성 / 2: 비활성)
나이
고객 가치 (계산된 값)
이탈 여부 (0: 비이탈 / 1: 이탈)

```python
data.info()
```

    명목형 데이터를 표현하는 컬럼들
    (요금제, 상태 등)은 각각 1과 0, 혹은
    1과 2로 이루어져 있기 때문에
    **머신러닝 분석에 직접적으로
    활용할 수 있는 상태**

![Untitled](img/Untitled%206.png)

![Untitled](img/Untitled%207.png)

    분석 타겟인 **Churn** 열의 분포도는 약 85:15로
    매우 불균형하게 이루어짐
    👉🏻 모델 성능 평가시 평가 지표로 **accuracy**는
          **적합하지 않음**

# 📝 전처리

---

## ▶ 결측치, 이상치 탐지 및 처리

```python
data.isna().sum()
```

```python
data.rename(columns=
	{'Subscription  Length':'Subscription Length',
	 'Call  Failure':'Call Failure', 
	 'Charge  Amount':'Charge Amount', 
	 'Frequency of use':'Frequency of Use'}, 
	 inplace=True)
```

![Untitled](img/Untitled%208.png)

    결측치를 가진 컬럼은 없었으나 **데이터 분석의 용이성을 위해 컬럼 이름을 일관성**있게 변경

```python
data.drop(data[data['Frequency of Use'] < data['Distinct Called Numbers']].index, inplace=True)
```

  **총 통화 상대 수가 총 통화 횟수
  보다 많다는 것은 논리적으로
  오류**가 있기 때문에 비정상적
  데이터로 간주하고 삭제 처리

    **분산 값이 높은 컬럼**의 **이상치** 탐지를 위해 **boxplot 생성**

```python
data.var().round(2).sort_values(ascending=False)
```

![Untitled](img/Untitled%209.png)

![Untitled](img/Untitled%2010.png)

    · 각 컬럼 전체 데이터의 약 10%에 해당하는 이상치 👉🏻 **이상치 처리 시 데이터 왜곡 가능성 있음**
    · 소비재(서비스)의 특성상 2080 법칙이라고도 하는 **파레토 법칙**에 해당하는 이상치일 수 있다고
    판단 👉🏻 **대량 사용자로 추정되는 값을 제거하지 않고 분석 진행**

## ▶ 평균치 컬럼 생성

```python
# 평균 통화 시간
data['Seconds per Use'] = data.apply(lambda row: (row['Seconds of Use'] / row['Frequency of Use']).round(2) if row['Frequency of Use'] > 0 else 0, axis=1)

# 월 평균 통화 수 - Call per Month
data['Call per Month'] = data.apply(lambda row: (row['Frequency of Use'] / (row['Subscription Length'] if row['Subscription Length'] < 9 else 9)).round(2), axis=1)

# 월 평균 문자 수 - SMS per Month
data['SMS per Month'] = data.apply(lambda row: (row['Frequency of SMS'] / (row['Subscription Length'] if row['Subscription Length'] < 9 else 9)).round(2), axis=1)

# 월 평균 통화 시간 - SMS per Month
data['Seconds per Month'] = data.apply(lambda row: (row['Seconds of Use'] / (row['Subscription Length'] if row['Subscription Length'] < 9 else 9)).round(2), axis=1)
```

  9개월 동안 수집된 데이터에는 가입 기간이 **9개월 미만**인 사람들의 이용 데이터도 포함되어 있기
  때문에 사용량을 나타내는 컬럼들에서 **불균형한 누적 결과**가 나타날 수 있다고 판단
  **👉🏻 사용량을 월 단위로 나누어 표현한 월별 사용량 컬럼을 생성**

## ▶ 표준화 컬럼 생성

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Standard Seconds'] = 
									scaler.fit_transform(data['Seconds of Use'].values.reshape(-1, 1))
data['Standard Call Freq'] = 
									scaler.fit_transform(data['Frequency of Use'].values.reshape(-1, 1))
data['Standard SMS Freq'] = 
									scaler.fit_transform(data['Frequency of SMS'].values.reshape(-1, 1))

**data['Usage Index'] =
			data['Standard Seconds'] + data['Standard Call Freq'] + data['Standard SMS Freq']**

del data['Standard Seconds']
del data['Standard Call Freq']
del data['Standard SMS Freq']
```

  사용량을 나타내는 통화량, 통화 시간, 문자량는 각각 다른 분포를 가지고 있으나 **데이터의 의미가
  유사**하다고 판단
  **👉🏻 통화량, 통화 시간, 문자량을 표준화하여 더한 값으로 이용 지수 컬럼을 생성**

# 📊 EDA

---

## ▶ 상관관계 분석

```python
plt.figure(figsize=(21, 9))
sns.heatmap(data.corr(), vmax=1.0, vmin=-1.0, cmap='coolwarm', annot=True)
```

![Untitled](img/Untitled%2011.png)

![Untitled](img/Untitled%2012.png)

    음의 상관관계가 높은 **Usage Index** 열과 **Customer Value** 열, 양의 상관관계가 높은 **Complains** 열
    에 대하여 바이올린 플롯을 그려보면 아래와 같다.

```python
fig = plt.figure(figsize=(21, 7))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.set_xlabel('Churn')
ax1.set_ylabel('Usage Index')
sns.violinplot(data=data, x='Churn', y='Usage Index', ax=ax1, hue='Churn', palette='coolwarm')

ax2.set_xlabel('Churn')
ax2.set_ylabel('Customer Value')
sns.violinplot(data=data, x='Churn', y='Customer Value', ax=ax2, hue='Churn', palette='coolwarm')

ax2.set_xlabel('Churn')
ax2.set_ylabel('Complains')
sns.violinplot(data=data, x='Churn', y='Complains', ax=ax3, hue='Churn', palette='coolwarm')

plt.show()
```

![Untitled](img/Untitled%2013.png)

## ▶ 모델링에 사용할 데이터

```python
data.corr()['Churn'].abs().sort_values(ascending=False)
```

    Churn 열과의 상관계수의 절대값이 큰 데이터들 중
    의미적으로 중복되는 사용량 컬럼들을 **Usage Index**로 대체하고
    Churn 포함 총 **8개의 열**을 가진 데이터프레임으로 재정의

```python
data = data.loc[:, ['Complains', 'Charge Amount', 'Distinct Called Numbers', 'Tariff Plan', 'Status', 'Customer Value', 'Churn', 'Usage Index']]

data.corr()
```

![Untitled](img/Untitled%2014.png)

![Untitled](img/Untitled%2015.png)

# 📊 모델링

---

## ▶ 데이터셋 분리

1. **입력 데이터**와 **타겟 데이터**로 분리

```python
X = data.drop(columns=['Churn'])
y = data['Churn']
```

1. **훈련 데이터셋**과 **테스트 데이터셋**으로 분리

```python
# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
```

1. 훈련 데이터셋과 테스트 데이터셋의 **타겟 분포**가 적절한지 확인

```python
import matplotlib.pyplot as plt
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

train_target_df = pd.DataFrame({'Churn':y_train})
sns.countplot(data=train_target_df, x='Churn', hue='Churn', palette='coolwarm', ax=ax1)
ax1.set_title('Train Set Target')

test_target_df = pd.DataFrame({'Churn':y_test})
sns.countplot(data=test_target_df, x='Churn', hue='Churn', palette='coolwarm', ax=ax2)
ax2.set_title('Test Set Target')

plt.show()
```

![Untitled](img/Untitled%2016.png)

## ▶ 지도학습

    **로지스틱회귀
    가장 기본적인** 분류 알고리즘으로 특성별 가중치를 통해 **결과에 영향을 주는 특성을 명확히
    이해**할 수 있음

**    의사결정트리**
    **트리 구조의 모델**로 예측 과정이 **직관적**이며 **결과를 시각화**하여 이해하기 쉬움

**    랜덤포레스트**
    **여러 개의 결정 트리를 조합**하여 **과적합 문제를 해결**하면서 **성능을 높일 수 있어** 적은 양의
    데이터셋으로 분석을 진행하는 현재 프로젝트에 적합하다고 판단

    **그라디언트부스팅**
    분석을 위한 데이터셋의 크기가 크지 않아 **과적합이 우려**되나, **순차 학습(오차 보정)**을 통해
    **높은 성능을 제공**한다는 점에 초점을 두고 모델링 진행  

**    XGBoost
    그라디언트 부스팅의 향상된 버전**으로 고성능을 기대해볼 수 있음

**모델링 순서**

1. 하이퍼파라미터 그리드 설정
2. GridSearch를 통해 최적의 하이퍼파라미터 도출
3. 최적의 하이퍼파라미터로 모델 학습
4. 결과 도출 및 모델 평가

![Untitled](img/Untitled%2017.png)

![Untitled](img/Untitled%2018.png)

![Untitled](img/Untitled%2019.png)

## ▶ 비지도학습

    **K-Means 클러스터링**
    타겟값이 있는 데이터셋이지만 클러스터의 분포를 통해 **데이터의 패턴을 탐색**하여 탈퇴 회원과
    비탈퇴 회원 간의 상이한 패턴이 있는지 데이터의 구조를 분석해볼 의도

**모델링 순서**

1. 상관계수가 높은 데이터 스케일링
2. 최적의 클러스터 수 찾기 
3. PCA로 차원축소하여 시각화
4. 결과 데이터 확인

![**최적의 K를 찾기 위한 엘보우 그래프**
K값에 따른 클러스터의 응집도를 나타내고 있다.
위 그래프에서 K가 4, 5일 때 기울기가 크게 꺾인다. ](img/Untitled%2020.png)

**최적의 K를 찾기 위한 엘보우 그래프**
K값에 따른 클러스터의 응집도를 나타내고 있다.
위 그래프에서 K가 4, 5일 때 기울기가 크게 꺾인다. 

![Untitled](img/Untitled%2021.png)

![Untitled](img/Untitled%2022.png)

  클러스터 2, 3에서
  **모호한 경계**를 보임

![Untitled](img/Untitled%2023.png)

![Untitled](img/Untitled%2024.png)

![Untitled](img/Untitled%2025.png)

## ▶ 이탈 고객 분석에 가장 잘 맞는 모델 선정

    **XGBoost 채택**
    모델 평가 지표에서 좋은 점수를 받은 XGBoost를 최종 모델로 선정

![Untitled](img/Untitled%2026.png)

```python
# Best hyperparameters
{'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100}
```

**하이퍼파라미터 최적화 과정 시각화**

![Untitled](img/Untitled%2027.png)

![Untitled](img/Untitled%2028.png)

![Untitled](img/Untitled%2029.png)

![Untitled](img/Untitled%2030.png)

![Untitled](img/Untitled%2031.png)

![Untitled](img/Untitled%2032.png)

![Untitled](img/Untitled%2033.png)

## ▶ 이탈 고객 예측 결과

최종적으로 채택된 XGBoost 모델을 사용해
테스트 데이터셋에서 임의로 뽑은 고객 10명에 대하여 이탈 여부를 예측한 결과

![Untitled](img/Untitled%2034.png)

![**289번 고객의 정보**](img/Untitled%2035.png)

**289번 고객의 정보**

# 💡 결론

---

## ▶ 고객 이탈률 개선 방안

1. **고객 불만(Complains)과 이탈률(Churn)의 높은 상관관계**
 데이터 분석 결과, **고객 불만과 이탈률 사이에 높은 상관관계**가 확인됨
불만 고객 대응 방안으로, 고객 불만 접수 시 즉각적인 대응 시스템 마련/ 정기적인 고객 만족도 조사 및 피드백 반영/고객 지원 서비스 강화 및 교육 등 회사 차원에서 불만 고객에 대한 적극적인 대응이 필요함
2. **사용량(Call / SMS)과 이탈률(Churn)의 상관관계**
**신규 고객 유치를 목적으로 하는 마케팅 전략**에서 **통화량 위주의 전략**을 펼치거나,
통화량이 많은 고객에게 추가 혜택 제공/ 통화량 기반의 고객 맞춤형 서비스 제공 등의 마케팅으로 **통화량이 많은 잠재 고객층을 가입시킬 수 있도록 유도**해야 함

## ▶ 수집 데이터 품질 개선 방안

1. **고객 불만(Complains)과 이탈률(Churn)의 높은 상관관계**
현행상 **불만을 측정할 수 있는 지표가 한정적**
서비스 품질 관리를 위해 **불만 측정 지표의 다변화**가 필요함
불만의 강도, 종류(내용 세분화), 빈도, 처리 시간 등 측정 항목을 세분화하여 데이터를 수집함으로써 고객 불만을 효과적으로 관리하여 고객 이탈률을 더 효과적으로 측정하고 개선시킬 수 있을 것으로 기대함
2. **고객 가치(Customer Value)와 사용량(Call / SMS)의 상관관계**
현재 고객 가치는 SMS 사용량과 상관관계가 높다.
분석 결과, **SMS보다 통화량이 이탈률과 더 높은 상관관계가 있는 것**으로 나타난다.
따라서, **고객 가치를 평가할 때 통화량과 관련된 값의 가중치를 증가**시킬 것을 고려해야 함.