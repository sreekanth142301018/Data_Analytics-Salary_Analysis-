# Salary Prediction Insights Dashboard

## Overview


This project is a web-based data analytics dashboard built with **Flask**. It enables users to upload their own datasets, perform machine learning predictions, and analyze association rules using interactive visualizations.



**Key Highlights:**
- **Case Study:** Dynamically generate custom visualizations by plotting graphs for any columns you select, enabling flexible and interactive data exploration.
- **Association Rules:** Generate and analyze skill association rules to discover valuable skill combinations and salary boost opportunities.
- **Prediction:** Select a target, view feature correlations, train multiple regression models (Linear, Ridge, SVR, Random Forest), compare MSE/R², and predict target values using the best model.

---

## Features

- **Upload CSV files** for analysis
- **Data preprocessing** and visualization with [Plotly](https://plotly.com/python/)
- **Predictive modeling**:
  - Linear Regression
  - Ridge Regression
  - Support Vector Regression (SVR)
  - Random Forest
- **Association Rule Mining** using the Apriori algorithm ([mlxtend](http://rasbt.github.io/mlxtend/))
- **Interactive dashboard** for:
  - Exploratory data analysis (EDA)
  - Model training and evaluation (MSE, R²)
  - Skill/feature association discovery

- **Salary Boost Suggestions**: Get clear, actionable recommendations on which additional skills (single or multiple) can increase your expected salary, based on association rule mining.

---

## Tech Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Visualization**: Plotly
- **Association Rules**: mlxtend
- **Frontend**: Bootstrap, HTML/CSS

---

## Project Structure

```
Salary-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── static/
│   ├── style.css
│   └── imgs/
│       ├── casestudy.png
│       ├── load.png
│       ├── loaddata.png
│       ├── pred.jpg
│       └── rules.png
└── templates/
    ├── association_rules.html
    ├── base.html
    ├── casestudy.html
    ├── choose_option.html
    ├── home.html
    ├── prediction.html
    ├── prediction_input.html
    └── prediction_result.html
```


## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/Salary-Analysis.git
cd Salary-Analysis
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the application

```sh
python app.py
```

### 4. Open in your browser

Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the dashboard.

---


## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Plotly](https://plotly.com/)
- [mlxtend](http://rasbt.github.io/mlxtend/)
- [scikit-learn](https://scikit-learn.org/)

---

Feel free to fork, contribute, or open issues!