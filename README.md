# 📊 Smart Workforce Productivity Prediction

A Machine Learning project that predicts **workforce productivity scores** using employee/workforce-related features and **Linear Regression**.

The project includes data analysis and visualization using Pandas, Matplotlib, and Seaborn, followed by training a Linear Regression model and evaluating its performance using **Explained Variance Score** and **Mean Absolute Error (MAE)**.

---

## 🚀 Features

* 📂 Load and explore workforce productivity data
* 🔍 Analyze dataset columns and statistics
* 📊 Visualize relationships using Seaborn pair plots
* ✂️ Split data into training and testing sets
* 📈 Train a Linear Regression model
* 🔮 Predict productivity scores
* 📏 Evaluate model performance
* 📊 Calculate Explained Variance Score
* 📉 Calculate Mean Absolute Error (MAE)

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning

* Linear Regression
* Train/Test Split
* Explained Variance Score
* Mean Absolute Error

---

## 📂 Project Structure

```text
Smart-Workforce-Productivity/
│
├── smart_workforce_productivity.csv
├── productivity_prediction.py
└── README.md
```

---

## 📊 Dataset

The project uses:

```text
smart_workforce_productivity.csv
```

The dataset contains workforce-related features and a target column:

```text
productivity_score
```

The target variable represents the productivity score that the model attempts to predict.

---

## 🔍 Data Exploration

The dataset is loaded using Pandas:

```python
df = pd.read_csv("smart_workforce_productivity.csv")
```

Basic information is then explored using:

```python
df.head()
df.columns
df.describe()
```

The target column is separated from the input features:

```python
x = df.drop("productivity_score", axis=1)

y = df["productivity_score"]
```

Where:

* `x` = input features
* `y` = productivity score / target

---

## 📊 Data Visualization

A Seaborn pair plot is used to visualize relationships between the different variables:

```python
sns.pairplot(df)
plt.show()
```

This helps explore relationships and patterns between workforce features and productivity scores.

---

## ✂️ Train/Test Split

The dataset is divided into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=5
)
```

The split uses:

* **80%** training data
* **20%** testing data

---

## 📈 Linear Regression

The Machine Learning model used in this project is **Linear Regression**.

```python
from sklearn.linear_model import LinearRegression

ln = LinearRegression()

ln.fit(x_train, y_train)
```

The model learns the relationship between the workforce features and the productivity score.

---

## 🔮 Predictions

After training, the model predicts productivity scores for the test data:

```python
predictions = ln.predict(x_test)
```

The predicted values are then compared with the actual productivity scores.

---

## 🧮 Model Parameters

The project also examines the Linear Regression model's:

### Intercept

```python
print(ln.intercept_)
```

### Coefficients

```python
print(ln.coef_)
```

The coefficients show how the input features contribute to the predicted productivity score within the fitted linear model.

---

## 📏 Model Evaluation

Two evaluation metrics are used.

### Explained Variance Score

```python
explained_variance_score(
    y_test,
    predictions
)
```

The Explained Variance Score measures how much of the variation in the target values is accounted for by the predictions. A value closer to **1** indicates that more of the variance is explained.

### Mean Absolute Error

```python
mean_absolute_error(
    y_test,
    predictions
)
```

MAE measures the average absolute difference between the actual and predicted productivity scores.

A lower MAE indicates smaller average prediction errors.

---

## 🔄 Machine Learning Workflow

```text
smart_workforce_productivity.csv
              ↓
       Load Dataset
              ↓
       Explore Data
              ↓
      Data Visualization
              ↓
    Separate X and Y
              ↓
       Train/Test Split
              ↓
     Linear Regression
              ↓
         Prediction
              ↓
       Model Evaluation
              ↓
 Explained Variance + MAE
```

---

## 🎓 Concepts Practiced

This project helped practice:

* Data loading
* Data exploration
* Pandas
* NumPy
* Data visualization
* Seaborn pair plots
* Feature/target separation
* Train/test splitting
* Linear Regression
* Model coefficients
* Model intercept
* Predictions
* Explained Variance Score
* Mean Absolute Error

---

## ▶️ How to Run

### 1. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Place the dataset

Make sure the following file is in the project folder:

```text
smart_workforce_productivity.csv
```

### 3. Run the Python file

```bash
python productivity_prediction.py
```

The program will display the dataset information, visualization, predictions, model parameters, and evaluation results.

---

## 👨‍💻 Author

**Abdul Rehman Hussain**

Python & Machine Learning Project
