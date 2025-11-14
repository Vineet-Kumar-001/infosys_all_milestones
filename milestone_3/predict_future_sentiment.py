import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import requests


# =======================
# 1️⃣ Load environment vars
# =======================
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
if not SLACK_WEBHOOK_URL:
    raise ValueError("❌ Slack webhook URL not found in .env file")


# =======================
# 2️⃣ Auto-detect CSV files in current folder
# =======================
folder_path = os.path.dirname(os.path.abspath(__file__))
file_paths = sorted(
    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
)

if not file_paths:
    raise FileNotFoundError("⚠️ No CSV files found in this folder!")

print(f"📂 Found {len(file_paths)} CSV files to process.")


# =======================
# 3️⃣ Seaborn Grid Visualization
# =======================
sns.set(style="whitegrid")
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()

avg_scores = []
base_time = datetime.now()
times = [base_time + timedelta(minutes=3 * i) for i in range(len(file_paths))]

for i, file in enumerate(file_paths):
    try:
        df = pd.read_csv(file)
        df["article_number"] = range(1, len(df) + 1)
        avg = df["sentiment_score"].mean()
        avg_scores.append(avg)

        sns.scatterplot(
            data=df,
            x="article_number",
            y="sentiment_score",
            hue="sentiment_label",
            palette={"Positive": "green", "Negative": "red", "Neutral": "gray"},
            s=80,
            edgecolor="black",
            alpha=0.9,
            ax=axes[i]
        )

        axes[i].set_title(f"Dataset {i+1}\nAvg Score: {avg:.3f}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Article Number")
        axes[i].set_ylabel("Sentiment Score")

    except Exception as e:
        print(f"⚠️ Could not process file {file}: {e}")

for j in range(len(file_paths), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("📊 Sentiment Score per Article — All Detected Datasets", fontsize=18, fontweight="bold", color="darkred")
plt.tight_layout(rect=[0, 0, 1, 0.97])
seaborn_chart_path = "sentiment_grid_chart.png"
plt.savefig(seaborn_chart_path)
plt.close()
print(f"✅ Seaborn chart saved as: {seaborn_chart_path}")


# =======================
# 4️⃣ Prophet Forecast
# =======================
prophet_df = pd.DataFrame({"ds": times, "y": avg_scores})
model = Prophet(seasonality_mode="additive")
model.fit(prophet_df)

future = model.make_future_dataframe(periods=5, freq="3min")
forecast = model.predict(future)

# Prophet next 5 predictions
future_forecast = forecast.tail(5)
prophet_preds = future_forecast["yhat"].values


# =======================
# 5️⃣ Polynomial Regression Forecast
# =======================
X = np.arange(len(avg_scores)).reshape(-1, 1)
y = np.array(avg_scores)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)
future_X = np.arange(len(avg_scores), len(avg_scores) + 5).reshape(-1, 1)
future_preds = poly_model.predict(poly.transform(future_X))



#  Plotly Forecast Visualization

actual_df = pd.DataFrame({"Index": np.arange(len(avg_scores)), "Sentiment_Score": avg_scores})
poly_df = pd.DataFrame({"Index": np.arange(len(avg_scores)), "Sentiment_Score": y_poly_pred})
prophet_df_future = pd.DataFrame({"Index": np.arange(len(avg_scores), len(avg_scores) + 5), "Sentiment_Score": prophet_preds})

fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(
    x=actual_df["Index"], y=actual_df["Sentiment_Score"],
    mode='lines+markers', name='Actual Data',
    line=dict(color='black', width=2)
))

# Polynomial regression line
fig.add_trace(go.Scatter(
    x=poly_df["Index"], y=poly_df["Sentiment_Score"],
    mode='lines', name='Polynomial Fit',
    line=dict(color='tomato', dash='dash')
))

# Polynomial future prediction
fig.add_trace(go.Scatter(
    x=np.arange(len(avg_scores), len(avg_scores) + 5),
    y=future_preds, mode='markers+lines', name='Polynomial Future',
    line=dict(color='green', width=2), marker=dict(symbol='diamond', size=10)
))

# Prophet future prediction
fig.add_trace(go.Scatter(
    x=prophet_df_future["Index"],
    y=prophet_df_future["Sentiment_Score"],
    mode='markers+lines',
    name='Prophet Future',
    line=dict(color='blue', width=2, dash='dot'),
    marker=dict(symbol='star', size=10)
))

fig.update_layout(
    title="🧠 Sentiment Forecasting — Prophet vs Polynomial Regression",
    xaxis_title="Data Index",
    yaxis_title="Average Sentiment Score",
    template="plotly_white",
    width=1000,
    height=600
)

plotly_chart_path = "sentiment_forecast_chart.png"
fig.write_image(plotly_chart_path)
print(f"✅ Plotly chart saved as: {plotly_chart_path}")


# =======================
# 7️⃣ Slack Notification
# =======================
message_lines = ["📈 *Predicted Next 5 Sentiment Scores (Polynomial):*"]
for val in future_preds:
    message_lines.append(f"• {val:.4f}")

message_lines.append("\n🔮 *Predicted Next 5 Sentiment Scores (Prophet):*")
for val in prophet_preds:
    message_lines.append(f"• {val:.4f}")

message = "\n".join(message_lines)
response = requests.post(SLACK_WEBHOOK_URL, json={"text": message})

if response.status_code == 200:
    print("✅ Predictions sent to Slack successfully!")
else:
    print(f"⚠️ Slack webhook error: {response.status_code} → {response.text}")
