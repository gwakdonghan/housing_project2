# 필수 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
file_path = './housingdata.csv'
data = pd.read_csv(file_path)
print(data)

# 결측치 확인
print(data.isnull().sum())

# 결측치를 각 열의 평균값으로 대체
data = data.fillna(data.mean())
# 결측치 확인 (모든 결측치가 0이 되어야 함)
print(data.isnull().sum())
# 데이터 크기 확인
print(data.shape)  # 데이터 크기는 (506, 14)로 유지됨

# IQR 경계값을 완화하여 이상치 제거
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
# 경계값 조정 (2.0 * IQR 사용)
data = data[~((data < (Q1 - 2.0 * IQR)) | (data > (Q3 + 2.0 * IQR))).any(axis=1)]
print(f"이상치 제거 후 데이터 크기: {data.shape}")

import seaborn as sns
import matplotlib.pyplot as plt

# 주요 특징선택을 위한 상관관계 높은 특성 출력
# 상관관계 행렬 계산
correlation_matrix = data.corr()

# 상관관계 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# MEDV와 상관관계가 높은 특성 출력
print(correlation_matrix['MEDV'].sort_values(ascending=False))

# 주요 특성 선택
selected_features = ['RM', 'LSTAT', 'PTRATIO', 'TAX', 'INDUS', 'NOX']
target = 'MEDV'

# 입력 데이터(X)와 목표 변수(y) 설정
X = data[selected_features]
y = data[target]

# 데이터 확인
print(f"선택된 특성 데이터:\n{X.head()}")
print(f"목표 변수 데이터:\n{y.head()}")

from sklearn.model_selection import train_test_split

# 데이터를 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 분할된 데이터 크기 확인
print(f"학습 데이터 크기: {X_train.shape}, {y_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}, {y_test.shape}")



# 여러 회귀 모델 비교 선형 회귀/의사결정나무/랜덤 포레스트 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 모델 리스트
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# 결과 저장 딕셔너리
results = {}

# 모델 학습 및 평가
for model_name, model in models.items():
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 테스트 데이터 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 결과 저장
    results[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "R²": r2
    }

# 결과 출력
for model_name, metrics in results.items():
    print(f"{model_name} - MAE: {metrics['MAE']:.2f}, MSE: {metrics['MSE']:.2f}, R²: {metrics['R²']:.2f}")


# 모델 성능 평가 / 결과 시각화 코드
import matplotlib.pyplot as plt
import pandas as pd

# 결과를 데이터프레임으로 변환
df_results = pd.DataFrame(results).T  # Transpose for easier handling

# 시각화: MAE, MSE, R²
metrics = ["MAE", "MSE", "R²"]

for metric in metrics:
    plt.figure(figsize=(8, 6))
    df_results[metric].plot(kind="bar", title=f"Model Comparison - {metric}", ylabel=metric)
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 최적의 모델을 선택하여 결과를 시각화
# Random Forest 모델로 예측
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# 실제값 vs 예측값 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.7, edgecolor=None)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted (Random Forest)", fontsize=16)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()


