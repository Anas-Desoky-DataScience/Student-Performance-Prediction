import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

warnings.filterwarnings('ignore')



# -------------------- Data Loading & Cleaning --------------------
df = pd.read_csv('StudentPerformanceFactors.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encoding
le = LabelEncoder()
cat_cols = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'Gender', 'School_Type']
for cat in cat_cols:
    df[cat] = le.fit_transform(df[cat])

low_mid_high_map = {'Low': 1, 'Medium': 2, 'High': 3}
low_mid_high_cats = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality']
for cat in low_mid_high_cats:
    df[cat] = df[cat].map(low_mid_high_map)

other_map = {
    'Positive': 1, 'Near': 1, 'High School': 1,
    'Neutral': 2, 'Moderate': 2, 'College': 2,
    'Negative': 3, 'Postgraduate': 3, 'Far': 3
}
other_cats = ['Peer_Influence', 'Distance_from_Home', 'Parental_Education_Level']
for cat in other_cats:
    df[cat] = df[cat].map(other_map)

# -------------------- Linear Regression --------------------
x = df[['Hours_Studied']]
y = df['Exam_Score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_train['Hours_Studied'], y=y_train, color='blue', label='Train data')
sns.scatterplot(x=x_test['Hours_Studied'], y=y_test, color='green', label='Test data')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression Model - Hours Studied vs Exam Score")
plt.legend()
plt.show()

# -------------------- Polynomial Regression --------------------
x2 = df[['Hours_Studied', 'Attendance']]
y2 = df['Exam_Score']

scaler = StandardScaler()
x2_scaled = scaler.fit_transform(x2)

poly_features = PolynomialFeatures(degree=2)
x2_poly = poly_features.fit_transform(x2_scaled)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2_poly, y2, test_size=0.3, random_state=42)

poly_reg = LinearRegression()
poly_reg.fit(x2_train, y2_train)
y2_pred = poly_reg.predict(x2_test)

# Metrics
mae2 = mean_absolute_error(y2_test, y2_pred)
mse2 = mean_squared_error(y2_test, y2_pred)
rmse2 = math.sqrt(mse2)
r2_2 = r2_score(y2_test, y2_pred)

print("\nPolynomial Regression Results:")
print("MAE:", mae2)
print("MSE:", mse2)
print("RMSE:", rmse2)
print("R²:", r2_2)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# --- Left: Linear Regression ---
sns.scatterplot(ax=axes[0], x=x_train['Hours_Studied'], y=y_train, color='blue', label='Train data')
sns.scatterplot(ax=axes[0], x=x_test['Hours_Studied'], y=y_test, color='green', label='Test data')
axes[0].plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
axes[0].set_xlabel("Hours Studied")
axes[0].set_ylabel("Exam Score")
axes[0].set_title("Linear Regression (Hours Studied)")
axes[0].legend()

# --- Right: Polynomial Regression ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Prepare mesh grid for surface
hours_range = np.linspace(x2['Hours_Studied'].min(), x2['Hours_Studied'].max(), 50)
att_range = np.linspace(x2['Attendance'].min(), x2['Attendance'].max(), 50)
H, A = np.meshgrid(hours_range, att_range)

# Predict values for the grid
grid = np.column_stack((H.ravel(), A.ravel()))
grid_scaled = scaler.transform(grid)
grid_poly = poly_features.transform(grid_scaled)
pred_grid = poly_reg.predict(grid_poly).reshape(H.shape)

# Plot regression surface
ax2.plot_surface(H, A, pred_grid, cmap=cm.coolwarm, alpha=0.6, linewidth=0, antialiased=True)

# Plot train and test points (unscaled)
train_original = df.loc[y2_train.index, ['Hours_Studied', 'Attendance', 'Exam_Score']]
test_original = df.loc[y2_test.index, ['Hours_Studied', 'Attendance', 'Exam_Score']]

ax2.scatter(train_original['Hours_Studied'], train_original['Attendance'], train_original['Exam_Score'],
           color='blue', label='Train Data', s=40)
ax2.scatter(test_original['Hours_Studied'], test_original['Attendance'], test_original['Exam_Score'],
           color='green', label='Test Data', s=40)

# Set labels and style
ax2.set_xlabel('Hours Studied')
ax2.set_ylabel('Attendance')
ax2.set_zlabel('Exam Score')
ax2.set_title('Polynomial Regression (3D): Hours Studied + Attendance')
ax2.view_init(elev=25, azim=120)
ax2.legend()

plt.tight_layout()
plt.show()