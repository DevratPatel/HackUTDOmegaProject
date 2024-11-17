from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
# Configure CORS to allow requests from your frontend domain
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],  # Add your frontend URL
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        print(data)
        return jsonify({"message": "Received"}), 200  # Add a response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)