import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
# use to select low cardinality categorical columns
categorical_divided_num = 10
randomForest_n_estimators = 250

X_full = pd.read_csv('../data/train.csv', index_col='Id')
X_test_full = pd.read_csv('../data/test.csv', index_col='Id')

# set the data's print to full-print
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, inplace=True, subset=['SalePrice'])
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# select categorical columns with relatively low cardinality(convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < categorical_divided_num and
                    X_train_full[cname].dtype == 'object']

# select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                  X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

print(X_train.head())

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

# Define model
model = RandomForestRegressor(n_estimators=randomForest_n_estimators, random_state=0)

# Bundle preprcessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

# scores = -1 * cross_val_score(clf, X_full, y, cv=5, scoring='neg_mean_absolute_error')

# print("Average MAE Score:", scores.mean())

preds_test = clf.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice':preds_test})
output.to_csv('submission.csv', index=False)