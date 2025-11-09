#ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# CSVファイルの読み込み
df = pd.read_csv('/content/drive/MyDrive/2025年3年/グループ/電力チーム/data/20160401_to_20220101_real1_1108.csv', encoding='Shift_JIS')

# データの構造確認
print(df.head())
print(df.columns)

#indexの作成
df = df.set_index(["ymdt","t"])

# 独立変数と被説明変数を定義
# 使用可能なDataFrame 'df_reset' から 'temperature' と 'real' を使用
df_reset = df.reset_index() # Add this line to create df_reset

X = df_reset['temperature']
y = df_reset['real']

# Combine X and y into a single DataFrame and drop rows with missing values
data_for_regression = pd.concat([X, y], axis=1).dropna()

X_cleaned = data_for_regression['temperature']
y_cleaned = data_for_regression['real']


# 二次関数での近似（多項式回帰）
X_sm = sm.add_constant(X_cleaned)
X_sm['temperature_sq'] = X_cleaned**2

# OLS (Ordinary Least Squares) モデルの推定
model = sm.OLS(y_cleaned, X_sm)
results = model.fit()

# フィッティング結果の表示
print(results.summary())

# 近似曲線を描画するためのxの値を作成
# データの最小値から最大値までの範囲で滑らかな曲線を描く
x_fit = np.linspace(X_cleaned.min(), X_cleaned.max(), 100)

# x_fit_sm を NumPy 配列として作成し、その後 DataFrame に変換してカラムを追加
x_fit_sm = sm.add_constant(x_fit)
x_fit_sm = pd.DataFrame(x_fit_sm, columns=['const', 'temperature'])
x_fit_sm['temperature_sq'] = x_fit**2
y_fit = results.predict(x_fit_sm)


# 元データと近似曲線をプロット
plt.figure(figsize=(10, 6))
plt.scatter(X_cleaned, y_cleaned, label='Original Data', alpha=0.5)
plt.plot(x_fit, y_fit, color='red', label=f'Quadratic Fit\ny = {results.params.iloc[2]:.4f} * temp^2 + {results.params.iloc[1]:.4f} * temp + {results.params.iloc[0]:.4f}')
plt.title('Quadratic Approximation of Temperature vs Real Power Consumption')
plt.xlabel('Temperature')
plt.ylabel('Real Power Consumption')
plt.legend()
plt.grid(True)
plt.show()

# OLS結果から係数を取得
const_coef = results.params.iloc[0]
temp_coef = results.params.iloc[1]
temp_sq_coef = results.params.iloc[2]

# Original quadratic equation
# 元の二次関数の式
print(f"二次関数の曲線式:")
print(f"y = {temp_sq_coef:.4f} * temperature^2 + {temp_coef:.4f} * temperature + {const_coef:.4f}")

# Complete the square: a*x^2 + b*x + c = a*(x + b/(2a))^2 + c - b^2/(4a)
# 平方完成の公式: a*x^2 + b*x + c = a*(x + b/(2a))^2 + c - b^2/(4a)
# Here, x is 'temperature', a is temp_sq_coef, b is temp_coef, c is const_coef
# ここで、xは'temperature'、aはtemp_sq_coef、bはtemp_coef、cはconst_coefです。

if temp_sq_coef != 0:
    h = -temp_coef / (2 * temp_sq_coef)
    k = const_coef - (temp_coef**2) / (4 * temp_sq_coef)
    print("\n平方完成した式:")
    print(f"y = {temp_sq_coef:.4f} * (temperature - ({h:.4f}))^2 + ({k:.4f})")
else:
    print("\n二次項の係数が0のため、平方完成はできません。")

import statsmodels.api as sm
import pandas as pd
import numpy as np

# 説明変数と目的変数を定義
# 'df_reset'から必要な列を選択
# 平日ダミーは、Saturday, Sunday, holiday_wd (平日祝日)がいずれも0の場合とする
# weekday変数は多重共線性の原因となるため、ここでは使用しない
# df_reset['weekday'] = ((df_reset['Saturday'] == 0) & (df_reset['Sunday'] == 0) & (df_reset['holiday_wd'] == 0)).astype(int)

# Calculate the temperature at the vertex from the previous quadratic fit
# 先ほどの二次関数近似から頂点の気温を計算
# From the previous quadratic fit: y = 4.3510 * (temperature - (16.4971))^2 + (2310.9550)
# The vertex temperature (h) was approximately 16.4971
vertex_temperature = 16.4971

# Create the new feature: absolute difference from the vertex temperature
# 新しい特徴量を作成: 頂点の気温との絶対差
df_reset['temp_diff_abs'] = np.abs(df_reset['temperature'] - vertex_temperature)

# Add the square of temp_diff_abs as a new feature
# temp_diff_abs の二乗を新しい特徴量として追加
df_reset['temp_diff_abs_sq'] = df_reset['temp_diff_abs']**2

# Convert 'ymdt' to datetime and extract the year
df_reset['ymdt'] = pd.to_datetime(df_reset['ymdt'], format='%Y/%m/%d %H:%M')
df_reset['year'] = df_reset['ymdt'].dt.year

# Create a dummy variable for the year 2020
# 2020年のダミー変数を作成
df_reset['covid'] = (df_reset['year'] == 2020).astype(int)


# 使用する説明変数 (X) と目的変数 (y)
# Saturday, Sunday, holiday_wd, temp_diff_abs, temp_diff_abs_sq に加えて covid を使用 (rainは削除)
# 説明変数 (X) と目的変数 (y) を定義
# Saturday, Sunday, holiday_wd, temp_diff_abs, temp_diff_abs_sq に加え、covid を使用 (rainは削除)
X = df_reset[['temp_diff_abs', 'temp_diff_abs_sq', 'Saturday', 'Sunday', 'covid']].copy() # Removed 'holiday_wd'
# SettingWithCopyWarningを避けるために .copy() を使用


y = df_reset['real']

# 説明変数に定数項を追加
X = sm.add_constant(X)

# OLS (Ordinary Least Squares) モデルを推定
model = sm.OLS(y, X)
results = model.fit()

# フィッティング結果を表示
print(results.summary())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure df_reset is available (assuming it's from previous cells)
# If not, you might need to recreate it or ensure the previous cell ran

# Get the actual 'real' values
actual_values = df_reset['real']

# Get the predicted values from the last regression model results
# Use the same X DataFrame that was used for fitting the last model
predicted_values = results.predict(X)

# Sort the data by temperature to get a smoother line for the predicted values
# This is helpful for visualizing the trend against temperature
plot_data = df_reset.copy()
plot_data['predicted_real'] = predicted_values
plot_data = plot_data.sort_values('temperature')

# Plot actual 'real' vs temperature
plt.figure(figsize=(10, 6))
plt.scatter(df_reset['temperature'], actual_values, label='Actual Real Power Consumption', alpha=0.5)

# Plot predicted 'real' vs temperature
# To get a smooth curve for the predicted values against temperature,
# we can create a range of temperature values and predict using the model.
# However, since the model includes categorical variables, predicting on
# a simple temperature range won't show the effect of weekday/holiday.
# A better way is to plot the predicted values corresponding to the actual temperatures,
# sorted by temperature.
plt.plot(plot_data['temperature'], plot_data['predicted_real'], color='red', label='Predicted Real Power Consumption (with additional features)')


plt.title('Actual vs Predicted Real Power Consumption vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Real Power Consumption')
plt.legend()
plt.grid(True)
plt.show()
