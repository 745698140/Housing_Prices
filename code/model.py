import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


iowa_file_path = '../data/train.csv'

home_data = pd.read_csv(iowa_file_path)

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

print(home_data.describe())

y = home_data.SalePrice

feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

X = home_data[feature_names]
#print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict((val_X))
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes:{:,.0f}".format(val_mae))

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X,train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("validationo MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X,y)

test_data_path = '../data/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data[feature_names]

test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id':test_data.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index = False)
