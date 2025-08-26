# Regression Problem – Data Preprocessing, Visualization & Modeling  

This project demonstrates solving a **regression problem** with proper **data preprocessing, outlier treatment, visualization, and regression modeling**.  

---

## 🔹 Steps Covered  

### 1. Data Handling  
- Loaded dataset using **Pandas**.  
- Basic exploration with:  
  ```python
  df.head(), df.info(), df.describe()
2. Numerical Operations

Applied NumPy for array manipulations and statistical measures.

Outlier Treatment (IQR Method)

Removed outliers using the Interquartile Range (IQR):
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['feature'] >= lower) & (df['feature'] <= upper)]

4. Data Visualization

Used Seaborn and Matplotlib for:

    Histograms & Distribution plots
    
    Boxplots for outlier detection
    
    Pairplots for feature-target relationships
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.boxplot(x=df['feature'])
    sns.pairplot(df)
    plt.show()
5. Regression Models Applied
   
🔹 Linear Regression

       from sklearn.linear_model import LinearRegression   
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
  🔹 Lasso Regression
  
     from sklearn.linear_model import Lasso
     lasso = Lasso(alpha=0.1)
     lasso.fit(X_train, y_train)
6. Model Evaluation

Evaluated models using metrics like:

    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)



Tools & Libraries

  Python
  
  NumPy
  
  Pandas
  
  Seaborn
  
  Matplotlib
  
  scikit-learn
