import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

def load_and_train_model():
    zomato_df = pd.read_csv(r'C:\devopspro2\app\zomato.csv', encoding='latin1')
    country_df = pd.read_excel(r'C:\devopspro2\app\Country-Code.xlsx')  # Fixed path here
    merged_df = pd.merge(zomato_df, country_df, on='Country Code', how='left')
    merged_df = merged_df[['Restaurant Name', 'City', 'Cuisines', 'Average Cost for two', 'Aggregate rating', 'Country']]
    merged_df.dropna(inplace=True)
    merged_df['Cuisines'] = merged_df['Cuisines'].apply(lambda x: x.split(',')[0].strip())
    merged_df.rename(columns={'Average Cost for two': 'Cost', 'Aggregate rating': 'Rating'}, inplace=True)

    cat = ['City', 'Cuisines', 'Country']
    num = ['Cost', 'Rating']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=10, random_state=42))
    ])

    features_df = merged_df[num + cat]
    pipeline.fit(features_df)
    merged_df['Cluster'] = pipeline.predict(features_df)

    return pipeline, merged_df

def recommend_restaurants(user_input, df, pipeline, top_n=5):
    user_df = pd.DataFrame([user_input])
    user_cluster = pipeline.predict(user_df)[0]
    cluster_df = df[df['Cluster'] == user_cluster].copy()

    cluster_features = cluster_df[['Cost', 'Rating', 'City', 'Cuisines', 'Country']]
    user_features = user_df[['Cost', 'Rating', 'City', 'Cuisines', 'Country']]

    cluster_transformed = pipeline.named_steps['preprocessor'].transform(cluster_features)
    user_transformed = pipeline.named_steps['preprocessor'].transform(user_features)

    if hasattr(cluster_transformed, "toarray"):
        cluster_transformed = cluster_transformed.toarray()
    if hasattr(user_transformed, "toarray"):
        user_transformed = user_transformed.toarray()

    distances = np.linalg.norm(cluster_transformed - user_transformed, axis=1)
    cluster_df['Distance'] = distances

    result = cluster_df.sort_values('Distance').head(top_n)
    return result[['Restaurant Name', 'City', 'Cuisines', 'Country', 'Cost', 'Rating']]
