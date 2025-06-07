
# Zomato Review Sentiment Analysis and Restaurant Clustering

## Project Overview

This project analyzes customer reviews from Zomato to extract sentiment insights and clusters restaurants based on various features such as location, cuisine, ratings, and sentiment scores. The goal is to understand customer satisfaction trends and identify meaningful groups of restaurants for better business insights and recommendations.

## Features

- Data preprocessing and cleaning for text reviews and restaurant metadata
- Sentiment analysis using NLP techniques
- Visualization of sentiment distribution across cuisines and locations
- Clustering restaurants with K-Means algorithm
- Interactive charts and detailed reports summarizing findings

## Dataset

This dataset contains comprehensive restaurant and review information designed for restaurant recommendation and analysis systems. The restaurant data includes essential business details such as restaurant names, web links, estimated per-person dining costs, and cuisine types served. Additionally, it captures operational information like restaurant timings and collection tags that align with popular food delivery platforms like Zomato categories.

The review data provides rich user-generated content with reviewer names, detailed review text, and numerical ratings. Each review entry includes valuable metadata such as the reviewer's profile information (number of reviews written and followers), timestamps indicating when reviews were posted, and the count of pictures accompanying each review. This dual-table structure enables comprehensive analysis of both restaurant characteristics and customer sentiment, making it suitable for building recommendation engines, performing sentiment analysis, or conducting market research on dining preferences and restaurant performance metrics.

Restaurant Data:

Restaurant name and web URL links
Estimated per-person dining cost
Cuisine types and categories served
Restaurant operating hours and timings
Collection tags aligned with Zomato platform categories

Review Data:

Reviewer identity and profile information
Complete review text content
Numerical rating scores
Reviewer engagement metrics (total reviews written, follower count)
Review timestamp and date information
Associated image count per review

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zomato-review-sentiment-clustering.git
   cd zomato-review-sentiment-clustering


2. Create a virtual environment and activate it (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

* Explore data and visualizations in the `notebooks/` folder.
* Run sentiment analysis and clustering scripts located in the `src/` folder:

  ```bash
  python src/sentiment_analysis.py
  python src/clustering.py
  ```
* Check the `reports/` directory for generated charts and summaries.

## Project Structure

```
zomato-review-sentiment-clustering/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for exploration and modeling
├── src/                   # Python scripts for preprocessing, analysis, clustering
├── models/                # Saved ML models
├── reports/               # Visualizations and reports
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
├── .gitignore             # Git ignore file
└── LICENSE                # License information
```

## Results

* Sentiment trends reveal customer satisfaction patterns.
* Clusters highlight restaurant groups based on cuisine, location, and ratings.
* Visualizations provide actionable insights for marketing and recommendations.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Created by Your Name*
*Contact: [your.email@example.com](mailto:your.email@example.com)*

```

---

If you want, I can help you generate the initial Python script for sentiment analysis or clustering next. Would you like that?
```
