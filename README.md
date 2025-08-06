# 📈 Insurance Charges Prediction with Linear Regression

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.2.3-orange.svg)
![NumPy](https://img.shields.io/badge/numpy-2.2.3-yellow.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-green.svg)
![Seaborn](https://img.shields.io/badge/seaborn-0.13.2-blue.svg)
![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.4-indigo.svg)
![Scikit-learn](https://img.shields.io/badge/scikit_learn-1.6.1-violet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive machine learning project that predicts individual medical insurance charges using linear regression techniques. This project demonstrates the complete data science workflow from exploratory data analysis to model evaluation and comparison.

## 📊 Project Overview

This project analyzes the Medical Cost Personal Dataset to build predictive models for insurance charges based on demographic and lifestyle factors. The analysis includes feature engineering, multiple modeling approaches, and thorough evaluation of model performance.

### 🎯 Objectives

- **Exploratory Data Analysis**: Understand feature distributions, relationships, and data quality
- **Feature Engineering**: Transform categorical variables and select optimal features
- **Model Development**: Build and compare linear regression models
- **Performance Evaluation**: Assess model accuracy and identify areas for improvement
- **Business Insights**: Provide actionable insights for insurance pricing strategies

## 📁 Project Structure

```
insurance-charges-prediction-linear-regression/
├── 📊 insurance.csv                                    # Dataset file
├── 📓 Insurance_Charges_Prediction_Linear_Regression.ipynb  # Main analysis notebook
├── 📄 Insurance_Charges_Prediction_Analysis_Report.pdf     # Detailed analysis report
├── 📋 requirements.txt                                 # Python dependencies
├── 📜 LICENSE                                          # MIT License
├── 📖 README.md                                        # This file
└── 🐍 .venv/                                          # Virtual environment
```

## 🛠️ Tools and Technologies

### Core Libraries
- **[pandas](https://pandas.pydata.org/) (2.2.3)**: Data manipulation and analysis
- **[NumPy](https://numpy.org/) (2.2.3)**: Numerical computing and array operations
- **[Matplotlib](https://matplotlib.org/) (3.10.0)**: Static, animated, and interactive visualizations
- **[Seaborn](https://seaborn.pydata.org/) (0.13.2)**: Statistical data visualization

### Machine Learning & Statistics
- **[scikit-learn](https://scikit-learn.org/) (1.6.1)**: Machine learning algorithms and tools
  - `train_test_split`: Data splitting for training and testing
  - `StandardScaler`: Feature standardization
  - `LassoCV`: Lasso regression with cross-validation
  - `mean_squared_error`, `mean_absolute_error`, `r2_score`: Evaluation metrics
- **[statsmodels](https://www.statsmodels.org/) (0.14.4)**: Statistical modeling and econometrics
  - `OLS`: Ordinary Least Squares regression
  - Statistical summaries and p-value analysis

### Development Environment
- **Python 3.11+**: Programming language
- **Jupyter Notebook**: Interactive development environment
- **Git**: Version control

## 📊 Dataset Information

**Source**: [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Size**: 1,338 records with 7 features

### Features Description

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `age` | Numerical | Age of the primary beneficiary | 18-64 years |
| `sex` | Categorical | Gender of the insurance contractor | male, female |
| `bmi` | Numerical | Body mass index | 15.96-53.13 |
| `children` | Numerical | Number of children/dependents | 0-5 |
| `smoker` | Categorical | Smoking status | yes, no |
| `region` | Categorical | Residential area in the US | northeast, northwest, southeast, southwest |
| `charges` | Numerical | **Target Variable** - Individual medical costs | $1,121.87-$63,770.43 |

## 🔍 Analysis Workflow

### 1. Data Loading & Initial Inspection
- Load dataset using pandas
- Examine data structure, types, and basic statistics
- Identify numerical vs categorical features

### 2. Data Cleaning & Preprocessing
- **Missing Values**: ✅ No missing values found
- **Duplicates**: ⚠️ Removed 1 duplicate record
- **Data Quality**: Verified data integrity and consistency

### 3. Exploratory Data Analysis (EDA)

#### Distribution Analysis
- **Age**: Higher representation of 18-20 year olds
- **BMI**: Fairly distributed across range
- **Children**: Expected linear distribution (0-5 children)
- **Charges**: Right-skewed with outliers indicating high-cost individuals

#### Correlation Analysis
```
Key Correlations with Charges:
├── Age: 0.30 (moderate positive)
├── BMI: 0.20 (weak-moderate positive)
└── Children: 0.067 (very weak)
```

#### Categorical Features Impact
- **Sex**: Minimal impact on charges (similar distributions)
- **Smoker**: **🔥 Strong predictor** - smokers have significantly higher charges
- **Region**: Weak predictor with minor regional differences

### 4. Feature Engineering & Selection

#### One-Hot Encoding
Applied to categorical variables:
- `sex` → `sex_male`
- `smoker` → `smoker_yes` 
- `region` → `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`

#### Statistical Feature Selection
Used backward elimination based on p-values (α = 0.05):

**Features Removed:**
- `sex_male` (p > 0.05)
- `region_northwest` (p > 0.05)
- `region_southwest` (p > 0.05)
- `region_southeast` (p > 0.05)

**Final Features Retained:**
- `age`, `bmi`, `children`, `smoker_yes`

### 5. Model Development & Training

#### Model 1: OLS Linear Regression (Statsmodels)
- **Algorithm**: Ordinary Least Squares
- **Features**: Backward elimination selected features
- **Train/Test Split**: 80/20
- **Library**: statsmodels

#### Model 2: Lasso Regression (Scikit-learn)
- **Algorithm**: Lasso with Cross-Validation
- **Features**: All encoded features (standardized)
- **Regularization**: Automatic alpha selection via CV
- **Library**: scikit-learn

## 📈 Model Performance

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.805 | Model explains 80.5% of variance (good performance) |
| **RMSE** | $5,993 | Average prediction error (magnifies larger errors) |
| **MAE** | $4,199 | Mean absolute error (treats all errors equally) |

### Key Findings

✅ **Strengths:**
- Strong explanatory power (R² = 0.805)
- Reasonable prediction accuracy for most cases
- Identifies key risk factors (smoking status, age, BMI)

⚠️ **Areas for Improvement:**
- Residual analysis shows distinct banding patterns
- Model averages between smoker/non-smoker groups
- Some larger prediction errors for high-cost cases

## 🔍 Model Diagnostics

### Residual Analysis
The residual plots reveal distinct horizontal bands, indicating that the model is averaging between different groups (smokers vs non-smokers) rather than making distinct predictions for each group.

### Recommendations for Improvement
1. **Separate Models**: Train individual models for smokers and non-smokers
2. **Outlier Handling**: Apply techniques to handle high-cost outliers
3. **Feature Engineering**: Create interaction terms (e.g., age × smoking status)
4. **Advanced Algorithms**: Consider polynomial regression or ensemble methods

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/insurance-charges-prediction-linear-regression.git
cd insurance-charges-prediction-linear-regression
```

2. **Create virtual environment:**
```bash
python -m venv .venv
```

3. **Activate virtual environment:**

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open the main analysis file:**
```
Insurance_Charges_Prediction_Linear_Regression.ipynb
```

3. **Run all cells** to reproduce the complete analysis

### Dependencies Installation

The `requirements.txt` file contains all necessary packages:

```text
pandas==2.2.3
numpy==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
statsmodels==0.14.4
scikit-learn==1.6.1
```

Install using:
```bash
pip install -r requirements.txt
```

## 📊 Key Visualizations

The notebook includes comprehensive visualizations:

1. **Distribution Plots**: Histograms of all numerical features
2. **Correlation Heatmap**: Feature correlation matrix
3. **Box Plots**: Categorical feature impact on charges
4. **Pair Plots**: Relationship exploration with smoking status highlighting
5. **Residual Plots**: Model diagnostic visualizations
6. **Actual vs Predicted**: Model performance visualization

## 🎯 Business Applications

### Insurance Pricing Strategy
- **Risk Assessment**: Identify high-risk customers (smokers, older age groups)
- **Premium Calculation**: Use model coefficients for pricing adjustments
- **Market Segmentation**: Different pricing strategies for different customer segments

### Key Insights for Insurers
1. **Smoking Status**: Most significant predictor (consider smoking cessation programs)
2. **Age Factor**: Gradual increase in costs with age
3. **BMI Impact**: Moderate correlation suggests health program opportunities
4. **Regional Differences**: Minimal impact on pricing decisions

## 📋 Project Limitations

1. **Dataset Scope**: US-based data may not generalize to other markets
2. **Feature Completeness**: Missing important factors (pre-existing conditions, income)
3. **Model Assumptions**: Linear relationship assumptions may be limiting
4. **Group Separation**: Single model struggles with distinct smoker/non-smoker groups

## 🔮 Future Enhancements

### Short-term Improvements
- [ ] Implement separate models for smokers and non-smokers
- [ ] Add polynomial features for non-linear relationships
- [ ] Apply advanced outlier detection and handling
- [ ] Cross-validation for robust model evaluation

### Long-term Extensions
- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] Deep learning approaches
- [ ] Time series analysis for cost trends
- [ ] Integration with real-world insurance data

## 📚 Additional Resources

- **Detailed Report**: See `Insurance_Charges_Prediction_Analysis_Report.pdf` for comprehensive analysis
- **Dataset Source**: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Statistical Methods**: [Statsmodels Documentation](https://www.statsmodels.org/)
- **Machine Learning**: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Project Author**: [Your Name]
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [@your-username](https://github.com/your-username)

## ⭐ Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Inspiration from various machine learning courses and tutorials
- Open source community for excellent libraries and tools

---

**⚡ Quick Start**: Jump to [Getting Started](#-getting-started) section to run the analysis immediately!

**📊 Results**: See [Model Performance](#-model-performance) for key findings and metrics!