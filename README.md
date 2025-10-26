# 🧠 Multiclass Classification — Random Forest vs Neural Network

This project applies and compares two **Machine Learning** approaches — a traditional **Random Forest Classifier** and a **Neural Network** built with **Keras** — for a multiclass classification task.  
It’s part of my ongoing journey in **Machine Learning and Deep Learning**, moving from classical algorithms to modern neural architectures.

---

## 📘 Project Overview

The goal is to predict the **class of samples** based on multiple input features.  
The dataset was preprocessed using standard scaling and label encoding, ensuring fair and consistent comparison between both models.

Two models were trained and evaluated:
- A **tuned Random Forest** optimized for balanced learning and interpretability.  
- A **fully connected Neural Network (Feedforward)** optimized with the Adam optimizer.

---

## ⚙️ Technologies Used

- **Python 3.x**  
- **NumPy**, **Pandas** → data manipulation  
- **Matplotlib**, **Seaborn** → data visualization  
- **Scikit-Learn** → preprocessing, model training, evaluation  
- **TensorFlow / Keras** → deep learning model  
- **Joblib** → model export  

---

## 🧩 Data Preprocessing Pipeline

1. **Feature Scaling**
   - Applied `StandardScaler` to normalize feature distributions.  

2. **Label Encoding**
   - Transformed categorical class labels into integer form using `LabelEncoder`.  

3. **Data Splitting**
   - Divided the dataset into training (80%) and testing (20%) sets for reliable evaluation.

4. **Model Training**
   - Implemented and trained both Random Forest and Neural Network models for fair comparison.

---

## 📊 Model Results

| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| **Random Forest** | 0.7875 | 0.7786 |
| **Neural Network** | **0.7875** | **0.7802** |

---

## 🧠 Model Performance Discussion

Both models achieved **very similar results**, with the **Neural Network slightly outperforming** the Random Forest in terms of F1-score (0.7802 vs. 0.7786).  

### 🔹 Random Forest Highlights
After tuning, the Random Forest used:
```python
{'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10,
 'min_samples_leaf': 4, 'class_weight': 'balanced'}
```

This setup provided stable results and handled class imbalance effectively.

### 🔹 Random Forest Highlights

The network architecture:
```python
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(num_classes, activation='softmax')
```

Compiled with:
```python
optimizer='adam'
loss='sparse_categorical_crossentropy'
metrics=['accuracy']
```

Despite its simplicity, this architecture matched the Random Forest’s performance, showing that even a basic feedforward network can perform competitively with good preprocessing.

---

## 🚀 Next Steps

- [ ] Add **Dropout**, **Batch Normalization**, and **Learning Rate Scheduling** for deeper experiments  
- [ ] Compare with **XGBoost** or **LightGBM** for potential performance boost  
- [ ] Perform **Feature Importance Analysis**  
- [ ] Export the trained **Neural Network model** for deployment  

---

## 🧾 Repository Structure

ML-Wine-Quality-Neural-Network-vs-Random-Forest/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── multiclass_comparison.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🧩 How to Run

git clone https://github.com/Ramy-Bahi/ML-Wine-Quality-Neural-Network-vs-Random-Forest.git
cd ML-Wine-Quality-Neural-Network-vs-Random-Forest
pip install -r requirements.txt

---

