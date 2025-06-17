🏠 House Price Prediction (SCT_ML_1)
Summary
A Machine Learning project that predicts housing prices based on engineered features like size, location, waterfront, renovation status, and more.
📊 Project Summary
- **Algorithm Used**: Linear Regression (with Ridge/Lasso options)
- **RMSE**: ~$183,000  
- **R² Score**: ~0.688  
- **Dataset Size**: 4,600 rows, 18 features
🔧 Features Engineered
- `sqft_living`  
- `price_per_sqft_area`  
- `waterfront`  
- `quality_index` (view + condition)  
- `lot_size_category`  
- `basement_present`  
- `recent_renovation`  
- `age_quality` (custom metric for newness + quality)
💡 Sample Predictions
| Type           | Predicted Price |
|----------------|-----------------|
| 2200 sqft home | $916,054        |
| Luxury home    | $2,721,699      |
| Budget home    | $347,597        |
📂 Files in this Repo
| File                   | Purpose                                |
|------------------------|----------------------------------------|
| `data.csv`             | Cleaned housing data                   |
| `house_price_prediction.py` | Main model training + prediction code |
| `feature_scaler.pkl`   | Saved MinMax scaler                    |
| `house_price_model.pkl`| Trained Linear Regression model        |
| `price_predictions.png`| Visualization of actual vs predicted   |
| `output.csv`           | Model evaluation output (optional)     |
| `predictions.csv`      | Exported prediction results            |
🚀 How to Run
1. Clone the repo:
```bash
git clone https://github.com/adiprabhu04/SCT_ML_1.git
cd SCT_ML_1
```
2. Install requirements (if needed):
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```
3. Run the model:
```bash
python house_price_prediction.py
```
🧠 Future Improvements
* Add SHAP/LIME explainability
* Web app interface (Streamlit or Flask)
* Cross-validation + hyperparameter tuning
* Real-time prediction API
📬 Contact
Made by [Aditya Prabhudessai](https://github.com/adiprabhu04)  
Feel free to fork, star, or open an issue!
