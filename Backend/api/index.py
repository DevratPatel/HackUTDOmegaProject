from flask import Flask, request, jsonify
from api import rag
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 

history = {}

global counter
counter = 0


@app.route('/postMessage', methods=['POST'])
def post_message():
    data = request.json
    message = data.get('message')
    convo_id = data.get('convoId', None)

    # Add message to history
    if convo_id is not None:
        messages = history[convo_id]
        messages.append(message)
    else:
        messages = [message]
        history[convo_id] = []
        history[convo_id].append(message)

    reply = rag.get_response(messages)
    suggestions = [rag.get_suggesstions(message)]

    # Create response
    response = {
        "reply": reply,
    }
    if suggestions:
        response["suggestions"] = suggestions

    # Include convoId only for the first response
    if convo_id is None:
        global counter
        counter += 1
        history[str(counter)] = messages
        response["convoId"] = str(counter)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
