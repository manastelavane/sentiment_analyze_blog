from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the Flask application
app = Flask(__name__)

# Download NLTK data (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

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
