import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
PRICE_LOWER = 75000
PRICE_UPPER = 3000000
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_clean_data():
    data = pd.read_csv('data.csv')
    print(f"Initial data shape: {data.shape}")
    print(f"Available columns: {list(data.columns)}")
    clean_data = data[
        (data['price'].between(PRICE_LOWER, PRICE_UPPER)) & 
        (data['bedrooms'] > 0) & 
        (data['bathrooms'] > 0) & 
        (data['sqft_living'] > 500)
    ].copy()
    print(f"Cleaned data shape: {clean_data.shape}")
    return clean_data

def engineer_features(data):
    df = data.copy()
    for col in ['view', 'condition', 'sqft_basement', 'yr_built']:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    df['age'] = 2023 - df['yr_built']
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

    df['statezip'] = df['statezip'].astype(str).str.strip()
    location_means = df.groupby('statezip').agg(
        mean_price=('price', 'mean'),
        mean_sqft=('sqft_living', 'mean')
    ).reset_index()
    location_means['pps_area'] = location_means['mean_price'] / location_means['mean_sqft']
    df = df.merge(location_means[['statezip', 'pps_area']], on='statezip', how='left')

    overall_pps = df['price'].mean() / df['sqft_living'].mean()
    df['price_per_sqft_area'] = df['pps_area'].fillna(overall_pps)

    df['lot_size_category'] = pd.cut(df['sqft_lot'], 
                                     bins=[0, 5000, 10000, 20000, np.inf],
                                     labels=[1, 2, 3, 4])
    df['basement_present'] = (df['sqft_basement'] > 0).astype(int)
    df['quality_index'] = (df['view'] + df['condition']) / 2
    df['recent_renovation'] = (df['yr_renovated'] > 2000).astype(int)
    df['age_quality'] = (100 - df['age']) * (df['quality_index'] / 5)
    df['lot_size_category'] = df['lot_size_category'].fillna(2)

    return df

def train_and_evaluate_model():
    data = load_and_clean_data()
    data = engineer_features(data)

    print("\nMissing values after engineering:")
    print(data.isnull().sum().sum())

    features = [
        'sqft_living',
        'price_per_sqft_area',
        'waterfront',
        'quality_index',
        'basement_present',
        'lot_size_category',
        'recent_renovation',
        'age_quality'
    ]
    
    X = data[features]
    y = data['price']

    # Convert categorical to numeric (lot_size_category)
    X['lot_size_category'] = X['lot_size_category'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use LinearRegression instead of Lasso
    model = LinearRegression(positive=True)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, PRICE_LOWER)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    unscaled_coef = model.coef_ / scaler.scale_
    intercept = model.intercept_

    print("\n=== Model Performance ===")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.4f}")

    print("\n=== Feature Impacts ===")
    for feature, coef in zip(features, unscaled_coef):
        print(f"{feature + ':':<25} ${coef:,.2f}")
    print(f"{'Base Price:':<25} ${intercept:,.2f}")

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(f"House Price Predictions (R²={r2:.3f}, RMSE=${rmse:,.0f})")
    plt.grid(True)
    plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("\nModel artifacts saved successfully")

    return model, scaler, features

def predict_house_price(sqft, area_price_sqft, waterfront=0, 
                        quality_index=2.5, basement_present=0, 
                        lot_size_category=2, recent_renovation=0, age_quality=50):
    try:
        model = joblib.load('house_price_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')

        house_data = {
            'sqft_living': [sqft],
            'price_per_sqft_area': [area_price_sqft],
            'waterfront': [waterfront],
            'quality_index': [quality_index],
            'basement_present': [basement_present],
            'lot_size_category': [float(lot_size_category)],
            'recent_renovation': [recent_renovation],
            'age_quality': [age_quality]
        }

        house_df = pd.DataFrame(house_data)
        scaled_data = scaler.transform(house_df)
        predicted_price = model.predict(scaled_data)[0]
        return max(predicted_price, PRICE_LOWER)

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

if __name__ == "__main__":
    model, scaler, features = train_and_evaluate_model()

    sample_price = predict_house_price(
        sqft=2200,
        area_price_sqft=350,
        waterfront=0,
        quality_index=3.0,
        basement_present=1,
        lot_size_category=2,
        recent_renovation=0,
        age_quality=70
    )
    if sample_price:
        print(f"\nSample Prediction for 2200 sqft house: ${sample_price:,.2f}")

    luxury_price = predict_house_price(
        sqft=4500,
        area_price_sqft=650,
        waterfront=1,
        quality_index=4.5,
        basement_present=1,
        lot_size_category=4,
        recent_renovation=1,
        age_quality=90
    )
    if luxury_price:
        print(f"Sample Prediction for luxury house: ${luxury_price:,.2f}")

    budget_price = predict_house_price(
        sqft=1200,
        area_price_sqft=250,
        waterfront=0,
        quality_index=2.0,
        basement_present=0,
        lot_size_category=1,
        recent_renovation=0,
        age_quality=40
    )
    if budget_price:
        print(f"Sample Prediction for budget house: ${budget_price:,.2f}")
