import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from textblob import TextBlob
import re
import spacy
import numpy as np
from collections import defaultdict

# Page configuration
st.set_page_config(page_title="Zomato Restaurant Clustering & Sentiment Analysis", layout="wide")
st.title("🍽️ Zomato Restaurant Clustering & Sentiment Analysis")
st.markdown("An interactive dashboard for analyzing Zomato restaurant data.")

# Cache spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable unused components
    except:
        st.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy_model()

# Sidebar for file upload and configuration
st.sidebar.header("Data Upload & Configuration")
restaurant_file = st.sidebar.file_uploader("Upload Names and Metadata.csv", type="csv")
reviews_file = st.sidebar.file_uploader("Upload Reviews.csv", type="csv")
max_rows = st.sidebar.slider("Max Rows to Process", 100, 10000, 1000, step=100)

@st.cache_data
def load_data(restaurant_file=None, reviews_file=None, max_rows=1000):
    try:
        if restaurant_file is not None:
            restaurant_data = pd.read_csv(restaurant_file)
        else:
            restaurant_data = pd.read_csv("Names and Metadata.csv")
        if reviews_file is not None:
            reviews = pd.read_csv(reviews_file)
        else:
            reviews = pd.read_csv("Reviews.csv")
        # Limit rows
        restaurant_data = restaurant_data.head(max_rows)
        reviews = reviews.head(max_rows)
        return restaurant_data, reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load data
restaurant_data, reviews = load_data(restaurant_file, reviews_file, max_rows)

if restaurant_data is not None and reviews is not None:
    # Preprocess data
    @st.cache_data
    def preprocess_data(restaurant_data, reviews, max_rows):
        with st.spinner("Preprocessing data..."):
            # Handle missing values
            restaurant_data = restaurant_data.drop("Collections", axis=1, errors="ignore")
            restaurant_data = restaurant_data.dropna()
            reviews = reviews.dropna()

            # Drop duplicates
            restaurant_data = restaurant_data.drop_duplicates()
            reviews = reviews.drop_duplicates()

            # Preprocess cuisines
            restaurant_data["Cuisines_list"] = restaurant_data.Cuisines.apply(lambda x: x.lower().replace(" ", "").split(","))

            # Preprocess cost
            restaurant_data["Cost"] = restaurant_data.Cost.apply(lambda x: str(x).replace(",", "")).astype(float)

            # Preprocess ratings
            reviews["Rating"] = pd.to_numeric(reviews.Rating, errors="coerce")
            numeric_count = reviews["Rating"].notna().sum()
            if numeric_count == 0:
                st.error("No numeric values in 'Rating' column. Using 0 as default.")
                reviews["Rating"] = reviews["Rating"].fillna(0)
            else:
                rating_mean = reviews["Rating"][reviews["Rating"].notna()].mean()
                reviews["Rating"] = reviews["Rating"].fillna(rating_mean)

            # Extract reviewer metadata
            def extract_follower_and_review_count(text):
                review_pattern = r"(\d+) Review"
                followers_pattern = r"(\d+) Follower"
                review_match = re.search(review_pattern, str(text))
                followers_match = re.search(followers_pattern, str(text))
                review = int(review_match.group(1)) if review_match else 0
                followers = int(followers_match.group(1)) if followers_match else 0
                return [review, followers]

            reviews[['prev_reviews_count', 'followers_count']] = reviews['Metadata'].apply(extract_follower_and_review_count).apply(pd.Series)
            reviews = reviews.drop('Metadata', axis=1)

            # Merge data
            merged_data = pd.merge(reviews, restaurant_data[["Name", "Cost", "Cuisines_list"]], 
                                  left_on='Restaurant', right_on='Name', how='inner')
            merged_data = merged_data.head(max_rows)

            # Convert Time to datetime
            reviews['Time'] = pd.to_datetime(reviews['Time'], errors='coerce')
            reviews['Month'] = reviews['Time'].dt.month
            reviews['DayOfWeek'] = reviews['Time'].dt.day_name()
            reviews['Hour'] = reviews['Time'].dt.hour

        return restaurant_data, reviews, merged_data

    restaurant_data, reviews, merged_data = preprocess_data(restaurant_data, reviews, max_rows)

    # Sidebar filters
    st.sidebar.subheader("Filter Options")
    cost_range = st.sidebar.slider("Cost Range", float(restaurant_data['Cost'].min()), 
                                  float(restaurant_data['Cost'].max()), 
                                  (float(restaurant_data['Cost'].min()), float(restaurant_data['Cost'].max())))
    
    cuisines = set(c for clist in restaurant_data['Cuisines_list'] for c in clist)
    selected_cuisines = st.sidebar.multiselect("Select Cuisines", options=sorted(cuisines), default=[])
    
    rating_range = st.sidebar.slider("Rating Range", float(reviews['Rating'].min()), 
                                    float(reviews['Rating'].max()), 
                                    (float(reviews['Rating'].min()), float(reviews['Rating'].max())))

    # Filter data
    filtered_restaurants = restaurant_data[
        (restaurant_data['Cost'].between(cost_range[0], cost_range[1])) &
        (restaurant_data['Cuisines_list'].apply(lambda x: any(c in x for c in selected_cuisines) if selected_cuisines else True))
    ]
    filtered_reviews = reviews[reviews['Rating'].between(rating_range[0], rating_range[1])]
    filtered_merged = merged_data[
        (merged_data['Cost'].between(cost_range[0], cost_range[1])) &
        (merged_data['Rating'].between(rating_range[0], rating_range[1])) &
        (merged_data['Cuisines_list'].apply(lambda x: any(c in x for c in selected_cuisines) if selected_cuisines else True))
    ]

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Sentiment Analysis", "Clustering"])

    with tab1:
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Restaurant Data")
            st.dataframe(filtered_restaurants[['Name', 'Cost', 'Cuisines']].head())
            st.write(f"Total Restaurants: {len(filtered_restaurants)}")
        
        with col2:
            st.subheader("Reviews Data")
            st.dataframe(filtered_reviews[['Reviewer', 'Rating', 'Review']].head())
            st.write(f"Total Reviews: {len(filtered_reviews)}")

        # Cuisine Distribution
        st.subheader("Top Cuisines")
        all_cuisines = [c for clist in filtered_restaurants['Cuisines_list'] for c in clist]
        cuisine_counts = pd.Series(all_cuisines).value_counts().head(10)
        fig = px.bar(x=cuisine_counts.values, y=cuisine_counts.index, orientation='h',
                     labels={'x': 'Number of Restaurants', 'y': 'Cuisine'},
                     title="Top 10 Cuisines")
        st.plotly_chart(fig, use_container_width=True)

        # Cost Distribution
        st.subheader("Cost Distribution")
        fig = px.histogram(filtered_restaurants, x='Cost', nbins=20, title="Restaurant Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Rating Trends
        st.subheader("Rating Trends")
        col1, col2, col3 = st.columns(3)
        monthly_avg = filtered_reviews.groupby('Month')['Rating'].mean()
        weekly_avg = filtered_reviews.groupby('DayOfWeek')['Rating'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        hourly_avg = filtered_reviews.groupby('Hour')['Rating'].mean()

        with col1:
            fig = px.line(x=monthly_avg.index, y=monthly_avg.values, labels={'x': 'Month', 'y': 'Avg Rating'},
                          title="Average Ratings by Month")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(x=weekly_avg.index, y=weekly_avg.values, labels={'x': 'Day', 'y': 'Avg Rating'},
                          title="Average Ratings by Day")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.line(x=hourly_avg.index, y=hourly_avg.values, labels={'x': 'Hour', 'y': 'Avg Rating'},
                          title="Average Ratings by Hour")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Sentiment Analysis")
        if st.button("Run Sentiment Analysis"):
            with st.spinner("Running sentiment analysis..."):
                # Clean reviews
                def clean_string(text):
                    return re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|\d+", "", str(text).lower())
                
                filtered_reviews['Review Cleaned'] = filtered_reviews['Review'].apply(clean_string)
                
                # Sentiment analysis
                filtered_reviews['Sentiment'] = filtered_reviews['Review Cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)
                filtered_reviews['Sentiment_Type'] = filtered_reviews['Sentiment'].apply(
                    lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                sentiment_counts = filtered_reviews['Sentiment_Type'].value_counts()
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)

                # Food entities
                if nlp:
                    @st.cache_data
                    def extract_food_entities(reviews):
                        return reviews['Review Cleaned'].apply(lambda x: [token.text for token in nlp(x) if token.pos_ == 'NOUN'])
                    
                    filtered_reviews['Food_Entities'] = extract_food_entities(filtered_reviews)

                    # Sentiment by food
                    st.subheader("Sentiment by Food Items")
                    food_sentiment_counts = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
                    for _, row in filtered_reviews.iterrows():
                        sentiment = row['Sentiment_Type']
                        sentiment_value = row['Sentiment']
                        for food_item in row['Food_Entities']:
                            food_sentiment_counts[food_item][sentiment] += sentiment_value
                    
                    food_sentiment_df = pd.DataFrame.from_dict(food_sentiment_counts, orient='index')
                    if not food_sentiment_df.empty:
                        threshold = food_sentiment_df['positive'].sort_values(ascending=False).iloc[19] if len(food_sentiment_df) > 20 else 0
                        top_sentiments = food_sentiment_df[food_sentiment_df['positive'] > threshold].sort_values(by='positive', ascending=False)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top_sentiments.index, y=top_sentiments['positive'], name='Positive', marker_color='green'))
                        fig.add_trace(go.Scatter(x=top_sentiments.index, y=-top_sentiments['negative'], name='Negative', mode='lines+markers', line=dict(color='red')))
                        fig.update_layout(title="Sentiment for Top 20 Food Items by Positive Sentiment",
                                         xaxis_title="Food Item", yaxis_title="Sentiment Value", xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Restaurant Clustering")
        if st.button("Run Clustering"):
            with st.spinner("Running clustering..."):
                # Feature engineering
                mlb = MultiLabelBinarizer()
                features = mlb.fit_transform(filtered_restaurants['Cuisines_list'])
                features_df = pd.DataFrame(features, columns=mlb.classes_, index=filtered_restaurants['Name'])
                features_df['Cost'] = filtered_restaurants.set_index('Name')['Cost']
                features_df['Avg_Ratings'] = filtered_merged.groupby('Restaurant')['Rating'].mean()
                features_df['Avg_Ratings'] = features_df['Avg_Ratings'].fillna(features_df['Avg_Ratings'].mean())
                
                # Select frequent cuisines
                selected_features = features_df.columns[features_df.sum(axis=0) > 7].tolist()
                if not selected_features:
                    st.warning("No cuisines appear frequently enough for clustering. Try increasing the dataset size.")
                else:
                    X = StandardScaler().fit_transform(features_df[selected_features])

                    # Clustering
                    n_clusters = st.slider("Number of Clusters", 2, min(15, len(X)), 7)
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                    features_df['Cluster'] = kmeans.fit_predict(X)

                    # Cluster results
                    results_df = features_df[selected_features + ['Cluster']]
                    results_df_grouped = results_df.groupby('Cluster').sum()
                    results_df_grouped[['Cost', 'Avg_Ratings']] = features_df[['Cost', 'Avg_Ratings', 'Cluster']].groupby('Cluster').mean()

                    st.subheader("Cluster Summary")
                    st.dataframe(results_df_grouped)

                    # Cuisine distribution heatmap
                    st.subheader("Cuisine Distribution Across Clusters")
                    cuisine_data = results_df_grouped.drop(['Cost', 'Avg_Ratings'], axis=1, errors='ignore')
                    fig = px.imshow(cuisine_data.T, title="Heatmap of Cuisine Distribution Across Clusters",
                                    labels=dict(x="Cluster", y="Cuisine"))
                    st.plotly_chart(fig, use_container_width=True)

                    # Cost vs Ratings scatter
                    st.subheader("Cost vs Average Ratings by Cluster")
                    fig = px.scatter(results_df_grouped.reset_index(), x='Cost', y='Avg_Ratings', color='Cluster',
                                     text='Cluster', title="Cost vs Average Ratings by Cluster")
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)

                    # Ratings to Cost Ratio
                    st.subheader("Ratings to Cost Ratio")
                    ratio = (results_df_grouped['Avg_Ratings'] / results_df_grouped['Cost']).sort_values(ascending=False)
                    fig = px.bar(x=ratio.index.astype(str), y=ratio.values, title="Ratings to Average Cost Ratio by Cluster",
                                 labels={'x': 'Cluster', 'y': 'Ratio'})
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Please upload both CSV files to proceed.")