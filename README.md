# 🌍 Global Life Expectancy Prediction by Country  
### Interactive Dashboard for Historical Analysis and Machine Learning Forecasting

This project is a Streamlit-based web application that allows users to explore and predict **life expectancy across countries worldwide** using key health and socio-economic indicators from the WHO dataset.

The application provides:

- Historical life expectancy visualization (2000–2015)
- Machine learning prediction for future years (2016+)
- Interactive world map with country + province boundaries
- User-friendly prediction form grouped by feature impact

---

## 📌 Project Motivation

Life expectancy is one of the most important indicators of national health and development.  
It reflects the combined influence of:

- Education
- Economic conditions
- Disease prevalence
- Child mortality
- Healthcare quality

This project aims to build a predictive model that can estimate life expectancy based on these indicators, while also providing an intuitive geographic dashboard for global exploration.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Provide an interactive dashboard to explore global life expectancy data.
2. Display actual recorded life expectancy values for historical years.
3. Build a machine learning model capable of predicting life expectancy for future years.
4. Allow users to simulate indicator changes and observe prediction outcomes.
5. Integrate spatial visualization through country boundary mapping.

---

## 🚀 Key Features

---

### ✅ 1. Historical Life Expectancy (2000–2015)

For years between **2000 and 2015**, the app displays the actual observed life expectancy value from the WHO dataset (if available).

Example output:

- Country: Afghanistan  
- Year: 2015  
- Actual life expectancy: 62.3 years  

---

### 🔮 2. Future Life Expectancy Prediction (2016+)

For years **2016 and beyond**, the app switches into prediction mode:

- A trained Lasso Regression model is loaded
- Users input indicator values
- The model predicts expected life expectancy

The form is auto-filled using the latest available record for the selected country.

---

### 🗺️ 3. Interactive World Map Visualization

The dashboard includes a fully interactive map built with:

- **Leafmap**
- **Folium**
- **Natural Earth boundaries**

Map functionality:

- World map shown first
- Selecting a country zooms automatically
- Optional province boundaries (Admin-1)
- Tooltips and popups for geographic information

---

### 📊 4. Indicator Grouping by Feature Impact

To improve interpretability, prediction indicators are grouped into three categories:

#### 🟢 High Positive Impact  
Variables that increase predicted life expectancy most strongly:

- Schooling  
- Alcohol  
- infant deaths  
- BMI  

#### 🔴 High Negative Impact  
Variables that decrease predicted life expectancy most strongly:

- HIV/AIDS  
- under-five deaths  
- Adult Mortality  
- Hepatitis B  

#### ⚪ Less Impact Variables  
Variables with smaller contribution:

- Diphtheria  
- Polio  
- percentage expenditure  
- GDP  
- Population  
- Measles  
- thinness 1–19 years  

---

## 📂 Dataset Information

The dataset used in this project is obtained directly from Kaggle:

- Dataset name: **Life Expectancy (WHO)**
- Kaggle source: `kumarajarshi/life-expectancy-who`
- File: `Life Expectancy Data.csv`

The dataset contains:

- 190+ countries
- Years from 2000 to 2015
- Health indicators + socio-economic metrics
- Target variable: Life expectancy

---

## 🧠 Machine Learning Model

---

### Model Type: Lasso Regression

This project uses a **Lasso Regression model**, which is a linear regression technique with L1 regularization.

Lasso is useful because it:

- Prevents overfitting
- Performs automatic feature selection
- Shrinks weak coefficients to zero

---

### Selected Features Used for Prediction

The final model uses the following 15 indicators:

```python
SELECTED_FEATURES = [
    "Schooling",
    "Alcohol",
    "infant deaths",
    "BMI",
    "Diphtheria",
    "Polio",
    "percentage expenditure",
    "GDP",
    "Population",
    "Measles",
    "thinness  1-19 years",
    "Hepatitis B",
    "Adult Mortality",
    "under-five deaths",
    "HIV/AIDS",
]
```

## 📈 Future Improvements

- Add confidence intervals for predictions
- Include Year trend modeling
- Deploy online via Streamlit Cloud
- Improve UI styling and filtering
- Add charts for indicator sensitivity analysis

## 📚 References

- WHO Global Health Observatory Dataset
- Kaggle Life Expectancy Dataset
- Natural Earth Boundary Data (Admin-0, Admin-1)

## 👨‍🎓 Author

- Developed by [Mr. Sophon THOY](https://github.com/sophon-pro)
- Developed as a Machine Learning and Data Science Final Project
- using WHO global health indicators and spatial visualization.
