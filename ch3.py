import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
from pathlib import Path

# Mean squared error warning - dismiss from terminal
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.metrics._regression"
)

# Get file prepped - minimal analysis
ROOT = Path(__file__).resolve().parent
file = ROOT / "statcast_2025.parquet"
df = pd.read_parquet(file)
df = df.dropna(subset=["hit_distance_sc", "launch_speed", "launch_angle"])

# Statistical Summary of Comparison Columns - only columns used here
cols = ["hit_distance_sc", "launch_speed", "launch_angle"]
print(f"{df[cols].describe().round(1)}")
y = df["hit_distance_sc"].values
X1 = df[["launch_speed"]].values
X2 = df[["launch_angle"]].values

# ----------------------------
# Correlation Eval - show that two independent variables will work together later
# ----------------------------
corr = df["launch_speed"].corr(df["launch_angle"])
print(f"Pearson correlation (EV vs LA): {corr:.4f}")

plot_df = df.sample(n=20000, random_state=42)

plt.figure(figsize=(6, 6))
plt.scatter(
    plot_df["launch_speed"],
    plot_df["launch_angle"],
    alpha=0.25,
    s=10
)

plt.xlabel("Exit Velocity")
plt.ylabel("Launch Angle")
plt.title(
    f"Launch Angle vs Exit Velocity\n"
    f"Pearson r = {corr:.3f}"
)

plt.axhline(0, color="gray", linewidth=1, alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------
# Grid for display - Exit Velocity
# ----------------------------
x1_grid = np.linspace(X1.min(), X1.max(), 300).reshape(-1, 1)

# Minimal sample of data for display
plot_df = df.sample(n=15000, random_state=42)
# ----------------------------
# A. Linear OLS
# ----------------------------
lin_ev = LinearRegression()
lin_ev.fit(X1, y)
y_lin = lin_ev.predict(x1_grid)

print("\n[EV] Linear OLS")
print(f"Intercept: {lin_ev.intercept_:.3f}")
print(f"Slope: {lin_ev.coef_[0]:.3f}")
print(f"R²: {lin_ev.score(X1, y):.4f}")

# ----------------------------
# B. Quadratic OLS
# ----------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X1_poly = poly.fit_transform(X1)
x1_grid_poly = poly.transform(x1_grid)

quad_ev = LinearRegression()
quad_ev.fit(X1_poly, y)
y_quad = quad_ev.predict(x1_grid_poly)

print("\n[EV] Quadratic OLS")
print(f"Coefficients: {quad_ev.coef_}")
print(f"R²: {quad_ev.score(X1_poly, y):.4f}")

# ----------------------------
# C. LOWESS
# ----------------------------
lowess_ev = lowess(
    y,
    X1.flatten(),
    frac=0.15,
    return_sorted=True
)

# LOWESS fitted values mapped back to original X - pseudo r2
lowess_ev_pred = np.interp(
    X1.flatten(),
    lowess_ev[:, 0],
    lowess_ev[:, 1]
)

lowess_ev_r2 = r2_score(y, lowess_ev_pred)

print("\n[EV] LOWESS")
print(f"Pseudo R²: {lowess_ev_r2:.4f}")

# ----------------------------
# D. kNN
# ----------------------------
knn_ev = KNeighborsRegressor(n_neighbors=50, weights="distance")
knn_ev.fit(X1, y)
y_knn = knn_ev.predict(x1_grid)

print("\n[EV] kNN Regression")
print(f"R²: {r2_score(y, knn_ev.predict(X1)):.4f}")

# ----------------------------
# Build plot
# ----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

# Linear
axs[0, 0].scatter(plot_df["launch_speed"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[0, 0].plot(x1_grid, y_lin, color="black", linewidth=3)
axs[0, 0].set_title("Linear OLS")

# Quadratic
axs[0, 1].scatter(plot_df["launch_speed"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[0, 1].plot(x1_grid, y_quad, color="green", linewidth=3)
axs[0, 1].set_title("Quadratic OLS")

# LOWESS
axs[1, 0].scatter(plot_df["launch_speed"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[1, 0].plot(lowess_ev[:, 0], lowess_ev[:, 1], color="orange", linewidth=3)
axs[1, 0].set_title("LOWESS")

# kNN
axs[1, 1].scatter(plot_df["launch_speed"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[1, 1].plot(x1_grid, y_knn, color="red", linewidth=3)
axs[1, 1].set_title("kNN Regression")

for ax in axs.flat:
    ax.set_xlabel("Exit Velocity")
    ax.set_ylabel("Hit Distance")

axs[0, 0].text(
    0.02, 0.95,
    f"R² = {lin_ev.score(X1, y):.3f}",
    transform=axs[0, 0].transAxes,
    va="top"
)

axs[0, 1].text(
    0.02, 0.95,
    f"R² = {quad_ev.score(X1_poly, y):.3f}",
    transform=axs[0, 1].transAxes,
    va="top"
)

axs[1, 0].text(
    0.02, 0.95,
    f"R² ≈ {lowess_ev_r2:.3f}",
    transform=axs[1, 0].transAxes,
    va="top"
)

axs[1, 1].text(
    0.02, 0.95,
    f"R² = {r2_score(y, knn_ev.predict(X1)):.3f}",
    transform=axs[1, 1].transAxes,
    va="top"
)

plt.suptitle("Hit Distance vs Exit Velocity — Model Comparison", fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------
# Grid for display - Launch Angle
# ----------------------------
x2_grid = np.linspace(X2.min(), X2.max(), 300).reshape(-1, 1)

# ----------------------------
# A. Linear OLS
# ----------------------------
lin_la = LinearRegression()
lin_la.fit(X2, y)
y_lin_la = lin_la.predict(x2_grid)

print("\n[LA] Linear OLS")
print(f"Intercept: {lin_la.intercept_:.3f}")
print(f"Slope: {lin_la.coef_[0]:.3f}")
print(f"R²: {lin_la.score(X2, y):.4f}")

# ----------------------------
# B. Quadratic OLS
# ----------------------------
X2_poly = poly.fit_transform(X2)
x2_grid_poly = poly.transform(x2_grid)

quad_la = LinearRegression()
quad_la.fit(X2_poly, y)
y_quad_la = quad_la.predict(x2_grid_poly)

print("\n[LA] Quadratic OLS")
print(f"Coefficients: {quad_la.coef_}")
print(f"R²: {quad_la.score(X2_poly, y):.4f}")

# ----------------------------
# C. LOWESS
# ----------------------------
lowess_la = lowess(
    y,
    X2.flatten(),
    frac=0.15,
    return_sorted=True
)

lowess_la_pred = np.interp(
    X2.flatten(),
    lowess_la[:, 0],
    lowess_la[:, 1]
)

lowess_la_r2 = r2_score(y, lowess_la_pred)

print("\n[LA] LOWESS")
print(f"Pseudo R²: {lowess_la_r2:.4f}")

# ----------------------------
# D. kNN
# ----------------------------
knn_la = KNeighborsRegressor(n_neighbors=50, weights="distance")
knn_la.fit(X2, y)
y_knn_la = knn_la.predict(x2_grid)

print("\n[LA] kNN Regression")
print(f"R²: {r2_score(y, knn_la.predict(X2)):.4f}")

# ----------------------------
# Build plot
# ----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

# Linear
axs[0, 0].scatter(plot_df["launch_angle"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[0, 0].plot(x2_grid, y_lin_la, color="black", linewidth=3)
axs[0, 0].set_title("Linear OLS")

# Quadratic
axs[0, 1].scatter(plot_df["launch_angle"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[0, 1].plot(x2_grid, y_quad_la, color="green", linewidth=3)
axs[0, 1].set_title("Quadratic OLS")

# LOWESS
axs[1, 0].scatter(plot_df["launch_angle"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[1, 0].plot(lowess_la[:, 0], lowess_la[:, 1], color="orange", linewidth=3)
axs[1, 0].set_title("LOWESS")

# kNN
axs[1, 1].scatter(plot_df["launch_angle"], plot_df["hit_distance_sc"], s=8, alpha=0.25)
axs[1, 1].plot(x2_grid, y_knn_la, color="red", linewidth=3)
axs[1, 1].set_title("kNN Regression")

for ax in axs.flat:
    ax.set_xlabel("Launch Angle")
    ax.set_ylabel("Hit Distance")
    ax.set_xlim(-10, 90)
    ax.set_ylim(-10, 500)

axs[0, 0].text(
    0.02, 0.95,
    f"R² = {lin_la.score(X2, y):.3f}",
    transform=axs[0, 0].transAxes,
    va="top"
)

axs[0, 1].text(
    0.02, 0.95,
    f"R² = {quad_la.score(X2_poly, y):.3f}",
    transform=axs[0, 1].transAxes,
    va="top"
)

axs[1, 0].text(
    0.02, 0.95,
    f"R² ≈ {lowess_la_r2:.3f}",
    transform=axs[1, 0].transAxes,
    va="top"
)

axs[1, 1].text(
    0.02, 0.95,
    f"R² = {r2_score(y, knn_la.predict(X2)):.3f}",
    transform=axs[1, 1].transAxes,
    va="top"
)

plt.suptitle("Hit Distance vs Launch Angle — Model Comparison", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Multivariate plot - data prep
X = df[["launch_speed", "launch_angle"]].values.astype(float)
y = df["hit_distance_sc"].values.astype(float)

plot_df = df.sample(n=25_000, random_state=42)
Xp = plot_df[["launch_speed", "launch_angle"]].values.astype(float)
yp = plot_df["hit_distance_sc"].values.astype(float)

# ----------------------------
# kNN smooth model for multivariate model
# ----------------------------
knn = KNeighborsRegressor(
    n_neighbors=150,
    weights="distance"
)
knn.fit(X, y)

y_hat = knn.predict(X)
r2 = r2_score(y, y_hat)
rmse = mean_squared_error(y, y_hat, squared=False)

print("Smooth kNN Surface")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f} ft")

# ----------------------------
# Grid for surface
# ----------------------------
ev_grid = np.linspace(Xp[:, 0].min(), Xp[:, 0].max(), 70)
la_grid = np.linspace(-10, 60, 70) 

EV, LA = np.meshgrid(ev_grid, la_grid)
grid = np.column_stack([EV.ravel(), LA.ravel()])

Z = knn.predict(grid).reshape(EV.shape)

# ----------------------------
# Density for coloring
# ----------------------------
kde = gaussian_kde(Xp.T)
density = kde(grid.T).reshape(EV.shape)
density /= density.max()

# ----------------------------
# Sample points + interpolation lines
# ----------------------------
sample_df = plot_df.sample(n=100, random_state=7)
sample_df = sample_df.loc[
    (sample_df["launch_angle"] >= -10) &
    (sample_df["launch_angle"] <= 60)
].copy()
Xs = sample_df[["launch_speed", "launch_angle"]].values
ys = sample_df["hit_distance_sc"].values
zs_hat = knn.predict(Xs)

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    EV, LA, Z,
    facecolors=plt.cm.viridis(density),
    linewidth=0,
    antialiased=True,
    alpha=0.9
)

# Interpolation lines
for i in range(len(Xs)):
    ax.plot(
        [Xs[i, 0], Xs[i, 0]],
        [Xs[i, 1], Xs[i, 1]],
        [zs_hat[i], ys[i]],
        color="black",
        linewidth=2,
        alpha=1
    )

# Sample points
ax.scatter(
    Xs[:, 0], Xs[:, 1], ys,
    color="black",
    s=35,
    alpha=1
)

# ----------------------------
# Axes + limits
# ----------------------------
ax.set_xlabel("Exit Velocity")
ax.set_ylabel("Launch Angle")
ax.set_zlabel("Hit Distance")

ax.set_ylim(-10, 60)
ax.set_zlim(0, 500)

ax.set_title(
    "Smooth kNN Regression Surface\n"
    "Expected Hit Distance by Exit Velocity and Launch Angle",
    pad=20
)

# ----------------------------
# Colorbar
# ----------------------------
mappable = plt.cm.ScalarMappable(cmap="viridis")
mappable.set_array(density)

cbar = plt.colorbar(
    mappable,
    ax=ax,
    shrink=0.6,
    pad=0.1
)
cbar.set_label("Relative Swing Density")

# ----------------------------
# Legend with metrics
# ----------------------------
legend_handle = Line2D(
    [0], [0],
    linestyle="none",
    marker="",
    label=f"kNN Surface\nR² = {r2:.3f}\nRMSE = {rmse:.2f} ft"
)

ax.legend(
    handles=[legend_handle],
    loc="upper left",
    frameon=True
)

plt.tight_layout()
plt.show()
