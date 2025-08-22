# 🪙 Gold Price Prediction using Machine Learning Algorithms

An end-to-end ML project to forecast gold prices using market indicators (SPX, GLD, USO, SLV, EUR/USD) and multiple regression models (Linear Regression, SVR, Decision Tree, Random Forest, KNN).:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

---

## 🔬 Background & Objectives

The goal is to build a reliable, data-driven framework for predicting gold price movements by learning patterns from historical market data. We leverage diverse algorithms to capture linear and non-linear relationships and compare their performance for practical forecasting.:contentReference[oaicite:2]{index=2}

**Why this matters**
- Helps investors and analysts assess risk and hedge positions.:contentReference[oaicite:3]{index=3}
- Turns multi-market signals (equities, commodities, FX) into actionable insights.:contentReference[oaicite:4]{index=4}

**Scope (high level)**
- Inputs (features): Date, S&P 500 (SPX), Gold (GLD), Oil (USO), Silver (SLV), EUR/USD.:contentReference[oaicite:5]{index=5}
- Models: Linear Regression, Support Vector Regression, Decision Tree, Random Forest, KNN.:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
- Outputs: Predicted values for forward periods based on learned historical relationships.:contentReference[oaicite:8]{index=8}


---

## 📊 Dataset Details

Our dataset is a **financial time series** with 🪙 gold prices and related market indicators.:contentReference[oaicite:0]{index=0}

**Shape:** 2,290 rows × 6 columns:contentReference[oaicite:1]{index=1}

**Features included:**
- 📅 **Date** → chronological record of observations  
- 📈 **S&P 500 Index (SPX)** → stock market performance indicator  
- 🪙 **Gold Prices (GLD)** → target variable for prediction  
- 🛢️ **United States Oil Fund (USO)** → energy commodity influence  
- 🪙 **Silver Prices (SLV)** → correlated precious metal  
- 💱 **EUR/USD** → exchange rate reflecting currency market movements:contentReference[oaicite:2]{index=2}

**Key Notes**
- Time range: 2008 → 2018 (training), predictions extended to 2019–2033:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}  
- Target variable: 🪙 **Gold Prices (GLD)**  
- Preprocessing steps included:  
  - Handling missing values & duplicates  
  - Outlier detection (IQR method)  
  - Scaling with `QuantileTransformer` for normalization:contentReference[oaicite:5]{index=5}  

The dataset serves as the foundation for uncovering relationships between **global markets, commodities, and currency exchange** that impact gold price dynamics.
