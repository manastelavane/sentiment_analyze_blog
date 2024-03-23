from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Initialize the Flask application
app = Flask(__name__)
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
print(nltk_data_dir)
# Download NLTK data (if not already downloaded)
if not os.path.exists(nltk_data_dir):
    nltk.download('vader_lexicon', download_dir=os.path.join(nltk_data_dir, 'sentiment', 'vader_lexicon.zip'))

# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Define a route for the GET request
@app.route('/', methods=['GET'])
def test_connection():
    return 'Connection is working!'

# Define the route for the sentiment analysis API
@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    # Get the blog post text from the request body
    blog_text = request.json.get('blog')

    # Perform sentiment analysis
    sentiment_scores = sid.polarity_scores(blog_text)

    # Determine sentiment
    if sentiment_scores['compound'] > 0:
        sentiment = 1  # Positive sentiment
    elif sentiment_scores['compound'] < 0:
        sentiment = -1  # Negative sentiment
    else:
        sentiment = 0  # Neutral sentiment

    # Return the sentiment score
    return jsonify({'sentiment': sentiment})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
