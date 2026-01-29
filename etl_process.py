import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv('sample_data.csv')

numeric_features = ['age', 'salary']
categorical_features = ['gender', 'city']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_transformed = preprocessor.fit_transform(data)

encoded_cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_features = numeric_features + list(encoded_cat_features)
processed_df = pd.DataFrame(data_transformed, columns=all_features)

processed_df.to_csv('processed_data.csv', index=False)
print("Processed Data saved as 'processed_data.csv'")