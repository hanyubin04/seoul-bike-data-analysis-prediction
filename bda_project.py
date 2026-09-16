# 1.라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 2.데이터 불러오기
# 따릉이 대여 데이터와 날씨 데이터를 불러옴
# 공공데이터는 한글이 포함되어 있어 cp949 인코딩을 사용함
bike1 = pd.read_csv("bike_1_6.csv", encoding="cp949")
bike2 = pd.read_csv("bike_7_12.csv", encoding="cp949")
weather = pd.read_csv("weather.csv", encoding="cp949")

# 3.데이터 전처리 및 병합
# 1~6월, 7~12월 따릉이 데이터를 하나로 합침
bike = pd.concat([bike1, bike2], ignore_index=True)

# 날짜 형식을 datetime 형식으로 변환
bike["대여일자"] = pd.to_datetime(bike["대여일자"])
weather["일시"] = pd.to_datetime(weather["일시"])

# 대여일자와 날씨 데이터의 일시를 기준으로 데이터 병합
df = pd.merge(
    bike,
    weather,
    left_on="대여일자",
    right_on="일시"
)

# 날짜 데이터에서 월과 요일 정보를 추출하여 파생 변수 생성
df["월"] = df["대여일자"].dt.month
df["요일"] = df["대여일자"].dt.dayofweek

# 4.한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 5.데이터 시각화
# 월별 평균 대여량
monthly = df.groupby("월")["대여건수"].mean()

plt.figure(figsize=(8,5))
monthly.plot(marker="o")

plt.title("월별 평균 따릉이 대여량")
plt.xlabel("월")
plt.ylabel("평균 대여건수")
plt.grid(True)

plt.show()

# 요일별 평균 대여량
weekday_avg = df.groupby("요일")["대여건수"].mean()

plt.figure(figsize=(8,5))
weekday_avg.plot(kind="bar")

plt.title("요일별 평균 따릉이 대여량")
plt.xlabel("요일")
plt.ylabel("평균 대여건수")

plt.xticks(
    [0,1,2,3,4,5,6],
    ["월","화","수","목","금","토","일"],
    rotation=0
)

plt.grid(True)
plt.show()

# 기온과 대여량 관계
plt.figure(figsize=(8,5))
plt.scatter(df["평균기온(°C)"], df["대여건수"])
plt.title("기온과 따릉이 대여량 관계")
plt.xlabel("평균기온(°C)")
plt.ylabel("대여건수")
plt.grid(True)
plt.show()

# 강수량과 대여량 관계
plt.figure(figsize=(8,5))
plt.scatter(df["일강수량(mm)"], df["대여건수"])
plt.title("강수량과 따릉이 대여량 관계")
plt.xlabel("일강수량(mm)")
plt.ylabel("대여건수")
plt.grid(True)
plt.show()

#상관관계 히트맵
corr_df = df[
    [
        "대여건수",
        "평균기온(°C)",
        "일강수량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "월",
        "요일"
    ]
]

corr = corr_df.corr()

plt.figure(figsize=(9,7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("따릉이 대여량과 날씨 변수의 상관관계")
plt.show()

# 6.예측 모델 생성
# 평균기온, 강수량, 풍속, 습도, 월, 요일을 입력값으로 사용하고 대여건수를 예측값으로 설정
# 입력 변수 x : 날씨 정보와 날짜 정보
# 출력 변수 y : 따릉이 대여건수
X = df[
    [
        "평균기온(°C)",
        "일강수량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "월",
        "요일"
    ]
]

y = df["대여건수"]

# 데이터 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest 회귀 모델 생성 후 학습시키기
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 7.모델 예측 및 성능 평가
# 테스트 데이터를 이용해 대여건수를 예측
pred = model.predict(X_test)

# 성능 확인
# R² Score를 사용해 모델 성능을 평가함
# 결과 약 0.835로, 대여량 변동 약 83.5% 설명 가능
r2 = r2_score(y_test, pred)
print("R² Score :", r2_score(y_test, pred))

# 변수 중요도 확인
importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, importance)
plt.title("변수 중요도")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.show()

# 실제값과 예측값 비교
plt.figure(figsize=(8,6))
plt.scatter(y_test, pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("실제 대여건수")
plt.ylabel("예측 대여건수")
plt.title("실제값과 예측값 비교")
plt.grid(True)
plt.show()

# 8.사용자 입력 기반 따릉이 대여량 예측
# 사용자가 직접 날씨 정보를 입력하면 예상 따릉이 대여건수를 출력
while True:
    print("\n===== 따릉이 대여량 예측 =====")

    temp = float(input("평균기온(°C): "))
    rain = float(input("일강수량(mm): "))
    wind = float(input("평균 풍속(m/s): "))
    humidity = float(input("평균 상대습도(%): "))
    month = int(input("월(1~12): "))
    input_weekday = int(input("요일(월=0, 화=1, 수=2, 목=3, 금=4, 토=5, 일=6): "))

    new_weather = pd.DataFrame({
        "평균기온(°C)": [temp],
        "일강수량(mm)": [rain],
        "평균 풍속(m/s)": [wind],
        "평균 상대습도(%)": [humidity],
        "월": [month],
        "요일": [input_weekday]
    })

    prediction = model.predict(new_weather)

    print(f"\n🚲 예상 따릉이 대여건수 : {int(prediction[0]):,} 건")

    again = input("\n다시 예측하시겠습니까? (y/n): ")
    if again.lower() != "y":
        break
