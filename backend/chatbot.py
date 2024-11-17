import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict
from tqdm import tqdm
import time

class ISPRecommendationSystem:
    def __init__(self, csv_path: str, samba_nova_key: str):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.csv_path = csv_path
        self.samba_nova_key = "4c08c7f6-63fe-4e36-bb86-dc042578a025"
        self.samba_nova_url = "https://api.sambanova.ai/v1/chat/completions"

        # Load and process data
        self.load_and_process_data()

        # Create product catalog context
        self.product_context = """[Your product context here]"""

    def load_and_process_data(self):
        """Load and process customer data"""
        print("Loading CSV data...")
        df = pd.read_csv(
            self.csv_path,
            dtype={
                'acct_id': str,
                'wireless_clients_count': 'Int32',
                'wired_clients_count': 'Int32',
                'extenders': 'Int32',
                'rx_avg_bps': 'float32',
                'tx_avg_bps': 'float32',
                'rssi_mean': 'float32'
            }
        )

        # Fill NA values
        df = df.fillna({
            'wireless_clients_count': 0,
            'wired_clients_count': 0,
            'extenders': 0,
            'city': 'Unknown',
            'state': 'Unknown',
            'whole_home_wifi': False,
            'wifi_security': False,
            'premium_tech_pro': False
        })

        print(f"Processing {len(df)} customer records...")
        self.vectors = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Create text representation
            text = f"""
            Customer Profile:
            Devices: {int(row['wireless_clients_count'])} wireless, {int(row['wired_clients_count'])} wired
            Network: {float(row['rx_avg_bps'])/1e6:.1f} Mbps down, {float(row['tx_avg_bps'])/1e6:.1f} Mbps up
            Signal: {float(row['rssi_mean'])} dBm
            Location: {row['city']}, {row['state']}
            """

            # Generate embedding
            embedding = self.embedding_model.encode(text)

            # Store vector with metadata
            self.vectors.append({
                'id': row['acct_id'],
                'vector': embedding,
                'metadata': {
                    'device_info': {
                        'wireless_count': int(row['wireless_clients_count']),
                        'wired_count': int(row['wired_clients_count']),
                        'total_devices': int(row['wireless_clients_count'] + row['wired_clients_count']),
                        'extenders': int(row['extenders'])
                    },
                    'network_performance': {
                        'download_mbps': float(row['rx_avg_bps'])/1e6,
                        'upload_mbps': float(row['tx_avg_bps'])/1e6,
                        'signal_strength': float(row['rssi_mean'])
                    },
                    'location': {
                        'city': str(row['city']),
                        'state': str(row['state'])
                    },
                    'services': {
                        'whole_home_wifi': bool(row['whole_home_wifi']),
                        'wifi_security': bool(row['wifi_security']),
                        'premium_tech_pro': bool(row['premium_tech_pro'])
                    }
                }
            })

        print(f"Created embeddings for {len(self.vectors)} customers")
        self.embeddings_matrix = np.array([v['vector'] for v in self.vectors])

    def analyze_scenario(self, scenario: str) -> str:
        """Analyze customer scenario and provide recommendations"""
        # Create scenario embedding
        scenario_embedding = self.embedding_model.encode(scenario)

        # Find similar profiles
        similarities = np.dot(self.embeddings_matrix, scenario_embedding)
        top_indices = np.argsort(similarities)[-5:][::-1]
        similar_profiles = [self.vectors[i]['metadata'] for i in top_indices]

        # Generate recommendations using LLM
        return self.generate_llm_response(scenario, similar_profiles)

    def generate_llm_response(self, scenario: str, similar_profiles: List[Dict]) -> str:
        """Generate recommendation using Samba Nova"""
        # Create prompt with statistics
        avg_devices = np.mean([p['device_info']['total_devices'] for p in similar_profiles])
        avg_download = np.mean([p['network_performance']['download_mbps'] for p in similar_profiles])

        prompt = f"""
        Analyze this customer scenario and provide internet service recommendations:

        Customer Scenario:
        {scenario}

        Similar Customer Patterns:
        - Average devices: {avg_devices:.1f}
        - Average download speed: {avg_download:.1f} Mbps
        - Customers with WiFi issues: {sum(1 for p in similar_profiles if p['network_performance']['signal_strength'] < -70)}

        Recommend appropriate internet plans and add-on services based on their needs.
        """

        try:
            response = requests.post(
                self.samba_nova_url,
                headers={
                    "Authorization": f"Bearer {self.samba_nova_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Meta-Llama-3.1-70B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are an ISP sales expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

def main():
    # Initialize system
    system = ISPRecommendationSystem(
        csv_path="current_customers-1(Powered by MaxAI).csv",
        samba_nova_key="ea5bee1c-32d8-4302-9d51-fed6523192c7"
    )

    # Test scenario
    scenario = """
    Family of 4 with multiple devices including:
    - Smart home devices
    - Gaming consoles
    - Work laptops
    Need reliable internet with good coverage and security features.
    """

    print("\nScenario:")
    print(scenario)
    print("\nRecommendations:")
    response = system.analyze_scenario(scenario)
    print(response)

# Rest of the code remains the same...
def chatbot_responser(system, input_text):

    # Test a scenario
    input_text = """
    Customer is looking for an affordable internet package with basic security features.
    They live in a small apartment and mainly use internet for browsing and video calls.
    """

    # print("\ninput_text:")
    # print(input_text)
    #print("\nRecommendations:")
    response = system.analyze_scenario(input_text)
    # print(response)
    return response

def main():
    # Initialize system
    system = ISPRecommendationSystem("current_customers-1(Powered by MaxAI).csv", "ea5bee1c-32d8-4302-9d51-fed6523192c7")

    # Test a scenario
    scenario = """
    Customer is looking for an affordable internet package with basic security features.
    They live in a small apartment and mainly use internet for browsing and video calls.
    """

    print("\nScenario:")
    print(scenario)
    print("\nRecommendations:")
    response = system.analyze_scenario(scenario)
    print(response)

if __name__ == "__main__":
    # main()
    # Initialize system
    system = ISPRecommendationSystem("current_customers-1(Powered by MaxAI).csv", "ea5bee1c-32d8-4302-9d51-fed6523192c7")
    print(chatbot_responser(system, "I have a new home that needs internet in Dallas. I am looking for a low cost bandwidth support with more security"))
    # print(chatbot_responser)