from flask import Flask, request, jsonify
from flask_cors import CORS
import re, rag, Levenshtein

# Defining stainless variables
app = Flask(__name__)
CORS(app)
global counter, history
history = {}
counter = 0


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)


@app.route('/postMessage', methods=['POST'])
def post_message():

    # Get data from request
    data = request.json
    message = data.get('message')
    global counter, history
    counter+=1
    history[str(counter)] = message

    # Get response from RAG
    reply = rag.get_recommendation(history)

    # Check for completed recommendations
    if "{" in reply:
        history = {}
        counter = 0

        products = {
            "Fiber 500": {
                "name": "Fiber 500",
                "price": "$45/mo",
                "description": "500Mbps Connection. Includes one standard WIFI router"
            },
            "Fiber 1 Gig": {
                "name": "Fiber 1 Gig",
                "price": "$65/mo",
                "description": "1Gbps Connection. Includes one standard WIFI router"
            },
            "Fiber 2 Gig": {
                "name": "Fiber 2 Gig",
                "price": "$99/mo",
                "description": "2Gbps Connection. Includes one upgraded WIFI router and one extender"
            },
            "Additional Extender": {
                "name": "Additional Extender",
                "price": "$5/mo per extender",
                "description": "Additional extender for above products"
            },
            "Fiber 5 Gig": {
                "name": "Fiber 5 Gig",
                "price": "$129/mo",
                "description": "5Gbps Connection. Includes one premium router"
            },
            "Fiber 7 Gig": {
                "name": "Fiber 7 Gig",
                "price": "$299/mo",
                "description": "7Gbps Connection. Includes one premium router and an extender at no charge"
            },
            "Whole Home WIFI": {
                "name": "Whole Home WIFI",
                "price": "$10.00/mo",
                "description": "Get the latest generation router with up to two additional extenders provided to Fiber 2 Gig speeds and below and 1 extender for 7 and 5 Gig to get a consistently strong Wi-Fi signal throughout the home."
            },
            "Unbreakable Wi-Fi": {
                "name": "Unbreakable Wi-Fi",
                "price": "$25.00/mo",
                "description": "Unbreakable Wi-Fi is an add-on service for Frontier Fiber Internet customers providing a backup internet during unexpected Frontier fiber network outages."
            },
            "Battery Back-up for Unbreakable Wi-Fi": {
                "name": "Battery Back-up for Unbreakable Wi-Fi",
                "price": "$130.00 one-time",
                "description": "Optional Battery Backup Unit (power pack) offers up to 4 hours of power during outages."
            },
            "Wi-Fi Security": {
                "name": "Wi-Fi Security",
                "price": "$5.00/mo",
                "description": "Advanced security managed via the app. Protects devices connected on the home network from malicious sites, scams, phishing."
            },
            "Wi-Fi Security Plus": {
                "name": "Wi-Fi Security Plus",
                "price": "$10.00/mo",
                "description": "Includes Wi-Fi Security, Multi-Device Security, VPN & Password Manager. Protects devices connected to home network and up to 3 devices while away."
            },
            "Total Shield": {
                "name": "Total Shield",
                "price": "$10.00/mo",
                "description": "Security (anti-virus) for up to 10 devices, including mobile devices. VPN protects your privacy and masks your online identity."
            },
            "My Premium Tech Pro": {
                "name": "My Premium Tech Pro",
                "price": "$10.00/mo",
                "description": "Provides additional tech support and services."
            },
            "Identity Protection": {
                "name": "Identity Protection",
                "price": "$10.00/mo",
                "description": "Includes personal information monitoring and up to $1M in identity theft insurance."
            },
            "Family Add-On": {
                "name": "Family Add-On",
                "price": "$5.00/mo each add'l user",
                "description": "Includes 1 additional user for Identity Protection."
            },
            "YouTube TV": {
                "name": "YouTube TV",
                "price": "$79.99/mo",
                "description": "100+ live channels with unlimited DVR storage and 3 simultaneous streams."
            }
        }

        pattern = r'"(.*?)"'
        matches = re.findall(pattern, reply)

        recs = []

        for item in matches:
            if "products" not in item:
                _, value = find_closest_word(item, products)
                recs.append(value)

        reply = {
            "reply": "Thank you for your information! Here are some recommendations for you:",
            "recommendations": recs
        }

        return jsonify(reply)

    counter+=1
    history[str(counter)] = reply

    # Create response
    response = {
        "reply": reply,
    }

    history = {}
    counter = 0
    return jsonify(response)

def find_closest_word(word, dictionary):
    closest_key = min(dictionary.keys(), key=lambda key: Levenshtein.distance(word, key))
    return closest_key, dictionary[closest_key]

if __name__ == '__main__':
    app.run(debug=True)
