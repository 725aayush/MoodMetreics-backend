from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load Hugging Face sentiment pipeline
classifier = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    result = classifier(text)[0]
    return jsonify({
        'label': result['label'],
        'score': round(result['score'], 2)
    })

@app.route('/')
def home():
    return "MoodMetrics API is running!"

if __name__ == '__main__':
    app.run(debug=True)
