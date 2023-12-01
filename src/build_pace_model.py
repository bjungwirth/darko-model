import pandas as pd 

df = pd.read_csv('data/game_paces.csv')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the XGBoost regressor model
model = xgb.XGBRegressor(n_estimators=1000,  # You can adjust the number of estimators
                         learning_rate=0.1,  # You can adjust the learning rate
                         max_depth=6)  # You can adjust the maximum depth of trees

# Fit the model to the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')