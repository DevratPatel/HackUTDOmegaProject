# %%
!pip install pinecone-client


# %%
from sentence_transformers import SentenceTransformer
import pinecone

# %%
from pinecone import Pinecone

pc = Pinecone(api_key="********-****-****-****-************")
index = pc.Index("quickstart") pcsk_2yuKhB_KD5q9be5wg7RZq46CY1HNStdkZHgXEPj1v7L6mriGitcj4iW8JkQnnbM558eqWv

# %%
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec

class ISPDataEmbedder:
    def __init__(self, pinecone_api_key: str):
        # Initialize single embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize Pinecone with new pattern
        self.pc = Pinecone(
            api_key=pinecone_api_key
        )

        # Check existing indexes
        self.list_indexes()

        # Get reference to existing index
        self.index = self.pc.Index("dataset-metadata")

    def list_indexes(self):
        """List all available indexes"""
        indexes = self.pc.list_indexes()
        print("Available indexes:", indexes)
        return indexes

    def embed_and_store(self, text: str, metadata: dict):
        """Embed text and store in Pinecone"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)

            # Store in Pinecone
            self.index.upsert(
                vectors=[(
                    metadata['id'],  # unique identifier
                    embedding.tolist(),
                    metadata
                )]
            )
            print(f"Successfully stored {metadata['id']}")
        except Exception as e:
            print(f"Error storing {metadata['id']}: {str(e)}")

def main():
    # Initialize with your API key
    api_key = "pcsk_2yuKhB_KD5q9be5wg7RZq46CY1HNStdkZHgXEPj1v7L6mriGitcj4iW8JkQnnbM558eqWv"
    embedder = ISPDataEmbedder(pinecone_api_key=api_key)

    # Store product info
    product_text = """
    Fiber 2 Gig 2Gbps Connection. Includes one upgraded WIFI router and one extender
    Price: $99/mo
    """
    embedder.embed_and_store(
        text=product_text,
        metadata={
            'id': 'product_fiber_2gig',
            'type': 'product',
            'name': 'Fiber 2 Gig',
            'price': 99.00
        }
    )

    # Store metric definition
    metric_text = """
    wireless_clients_count indicates the number of WIFI-connected devices
    on the customer network. Average is 12.04 devices.
    """
    embedder.embed_and_store(
        text=metric_text,
        metadata={
            'id': 'metric_wireless_clients',
            'type': 'metric_definition',
            'name': 'wireless_clients_count'
        }
    )

    # Test query to verify storage
    try:
        query_response = embedder.index.query(
            vector=[0.0] * 384,  # dummy vector
            top_k=5,
            include_metadata=True
        )
        print("\nQuery test results:", query_response)
    except Exception as e:
        print(f"Error querying index: {str(e)}")

if __name__ == "__main__":
    main()

# %%
import pandas as pd

# %%
df = pd.read_csv('/content/current_customers-1(Powered by MaxAI).csv')

# %%
df.head()

# %%
df.isnull().sum()

# %%
import pandas as pd
from typing import Dict, List
import requests
import json
import numpy as np

class CSVChatBot:
    def __init__(self, samba_nova_key: str, csv_path: str):
        """Initialize CSV chatbot"""
        self.api_key = "4c08c7f6-63fe-4e36-bb86-dc042578a025"
        self.base_url = "https://api.sambanova.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        print(f"Loading CSV from {csv_path}...")

        # Define dtypes for columns to avoid warnings
        dtypes = {
            'acct_id': str,
            'wireless_clients_count': float,
            'wired_clients_count': float,
            'extenders': float,
            'rx_avg_bps': float,
            'tx_avg_bps': float,
            'rssi_mean': float,
            'city': str,
            'state': str,
            'whole_home_wifi': bool,
            'wifi_security': bool,
            'wifi_security_plus': bool,
            'premium_tech_pro': bool
        }

        # Load CSV with specified dtypes and handle missing values
        self.df = pd.read_csv(
            csv_path,
            dtype=dtypes,
            na_values=['NaN', 'null', ''],
            low_memory=False
        )

        print(f"Loaded {len(self.df)} records")

        # Create initial context about the data
        self.data_context = self._create_data_context()

    def _create_data_context(self) -> str:
        """Create context about the CSV data"""
        try:
            # Calculate basic statistics safely
            wireless_stats = self.df['wireless_clients_count'].agg(['mean', 'median', 'min', 'max']).to_dict()
            wired_stats = self.df['wired_clients_count'].agg(['mean', 'median', 'min', 'max']).to_dict()

            context = f"""
            Dataset Overview:
            Total Records: {len(self.df)}

            Wireless Clients Statistics:
            - Mean: {wireless_stats['mean']:.2f} devices
            - Median: {wireless_stats['median']:.2f} devices
            - Range: {wireless_stats['min']:.0f} to {wireless_stats['max']:.0f} devices

            Wired Clients Statistics:
            - Mean: {wired_stats['mean']:.2f} devices
            - Median: {wired_stats['median']:.2f} devices
            - Range: {wired_stats['min']:.0f} to {wired_stats['max']:.0f} devices

            Service Adoption:
            - Whole Home WiFi: {self.df['whole_home_wifi'].sum()} customers
            - WiFi Security: {self.df['wifi_security'].sum()} customers

            Geographic Distribution:
            - States: {self.df['state'].nunique()} unique states
            - Cities: {self.df['city'].nunique()} unique cities
            """

            return context
        except Exception as e:
            print(f"Error creating data context: {str(e)}")
            return "Error creating data context"

    def chat(self, query: str) -> str:
        """Chat with the CSV data"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an ISP data analyst assistant. Use the following context about the data to answer questions:

                    {self.data_context}

                    When analyzing the data:
                    1. Provide specific numbers and statistics
                    2. Consider both technical metrics and business insights
                    3. Make recommendations when relevant
                    """
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            if "example" in query.lower() or "show" in query.lower():
                sample_data = self.df.head(5).to_string()
                messages[1]["content"] += f"\n\nSample Data:\n{sample_data}"

            payload = {
                "model": "Meta-Llama-3.1-70B-Instruct",
                "messages": messages,
                "temperature": 0.7,
                "stream": False  # Changed to False for simpler response handling
            }

            print("Sending request to Samba Nova...")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30  # Added timeout
            )

            # Print response for debugging
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")  # Print first 200 chars

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                return "No response generated"

        except requests.exceptions.RequestException as e:
            return f"API request error: {str(e)}"
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the dataset"""
        try:
            stats = {
                "total_records": len(self.df),
                "wireless_clients": {
                    "mean": float(self.df['wireless_clients_count'].mean()),
                    "median": float(self.df['wireless_clients_count'].median()),
                    "min": float(self.df['wireless_clients_count'].min()),
                    "max": float(self.df['wireless_clients_count'].max())
                },
                "service_adoption": {
                    "whole_home_wifi": int(self.df['whole_home_wifi'].sum()),
                    "wifi_security": int(self.df['wifi_security'].sum())
                },
                "geographic": {
                    "states": int(self.df['state'].nunique()),
                    "cities": int(self.df['city'].nunique())
                }
            }
            return stats
        except Exception as e:
            return {"error": str(e)}

def main():
    try:
        print("Initializing chatbot...")
        chatbot = CSVChatBot(
            samba_nova_key="ea5bee1c-32d8-4302-9d51-fed6523192c7",
            csv_path="current_customers.csv"
        )

        # Get basic stats first
        print("\nBasic Statistics:")
        stats = chatbot.get_basic_stats()
        print(json.dumps(stats, indent=2))

        # Test with a simple query
        test_queries = [
            "How many customers are in the dataset?",
            "What's the average number of wireless devices per customer?",
            "Which states have the most customers?"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            response = chatbot.chat(query)
            print(f"Response: {response}")

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

class ISPDataProcessor:
    def __init__(self, samba_nova_key: str, csv_path: str):
        self.api_key =  "4c08c7f6-63fe-4e36-bb86-dc042578a025"
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Load data with proper NA handling
        print("Loading CSV data...")
        self.df = self._load_csv_safely(csv_path)
        print(f"Loaded {len(self.df)} records")

    def _load_csv_safely(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with proper handling of data types and NA values"""
        # Define dtypes with explicit NA handling
        dtypes = {
            'acct_id': str,
            'extenders': 'Int64',
            'wireless_clients_count': 'Int64',
            'wired_clients_count': 'Int64',
            'rx_avg_bps': float,
            'tx_avg_bps': float,
            'rx_p95_bps': float,
            'tx_p95_bps': float,
            'rx_max_bps': float,
            'tx_max_bps': float,
            'rssi_mean': float,
            'rssi_median': 'Int64',
            'rssi_max': 'Int64',
            'rssi_min': 'Int64',
            'network_speed': str,
            'city': str,
            'state': str
        }

        # Boolean columns need special handling
        bool_columns = [
            'whole_home_wifi',
            'wifi_security',
            'wifi_security_plus',
            'premium_tech_pro',
            'identity_protection',
            'family_identity_protection',
            'total_shield',
            'youtube_tv'
        ]

        try:
            # First pass: Load with basic types
            df = pd.read_csv(
                csv_path,
                dtype={**dtypes, **{col: object for col in bool_columns}},
                na_values=['NaN', 'null', '', 'NA', 'None'],
                keep_default_na=True
            )

            # Handle boolean columns
            for col in bool_columns:
                # Convert to boolean with NA handling
                df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
                df[col] = df[col].fillna(False)  # Convert NaN to False

            # Fill NA values appropriately
            df['extenders'] = df['extenders'].fillna(0)
            df['wireless_clients_count'] = df['wireless_clients_count'].fillna(0)
            df['wired_clients_count'] = df['wired_clients_count'].fillna(0)

            # Fill network metrics with median values
            numeric_cols = ['rx_avg_bps', 'tx_avg_bps', 'rx_p95_bps', 'tx_p95_bps',
                          'rx_max_bps', 'tx_max_bps', 'rssi_mean', 'rssi_median',
                          'rssi_max', 'rssi_min']

            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

            # Fill categorical values
            df['network_speed'] = df['network_speed'].fillna('1000.0M')  # Most common speed
            df['city'] = df['city'].fillna('Unknown')
            df['state'] = df['state'].fillna('Unknown')

            # Verify no NA values remain
            na_counts = df.isna().sum()
            if na_counts.sum() > 0:
                print("Remaining NA counts:")
                print(na_counts[na_counts > 0])

            return df

        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise

    def process_and_store_embeddings(self, batch_size: int = 100):
        """Process and store embeddings in batches"""
        vectors = []
        total_records = len(self.df)

        for start_idx in tqdm(range(0, total_records, batch_size), desc="Processing customers"):
            batch = self.df.iloc[start_idx:start_idx + batch_size]

            for _, row in batch.iterrows():
                try:
                    # Create metadata
                    metadata = {
                        'customer_id': str(row['acct_id']),
                        'device_info': {
                            'wireless_count': int(row['wireless_clients_count']),
                            'wired_count': int(row['wired_clients_count']),
                            'extenders': int(row['extenders']),
                            'total_devices': int(row['wireless_clients_count'] + row['wired_clients_count'])
                        },
                        'network_performance': {
                            'download_mbps': float(row['rx_avg_bps']) / 1_000_000,
                            'upload_mbps': float(row['tx_avg_bps']) / 1_000_000,
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

                    # Create text for embedding
                    text = self._create_embedding_text(row)
                    embedding = self.embedding_model.encode(text)

                    vectors.append({
                        'id': str(row['acct_id']),
                        'vector': embedding.tolist(),
                        'metadata': metadata
                    })

                except Exception as e:
                    print(f"Error processing row {row['acct_id']}: {str(e)}")
                    continue

        return vectors

    def _create_embedding_text(self, row: pd.Series) -> str:
        """Create text representation for embedding"""
        return f"""
        Customer Profile:
        Network Configuration:
        - Wireless Devices: {int(row['wireless_clients_count'])}
        - Wired Devices: {int(row['wired_clients_count'])}
        - WiFi Extenders: {int(row['extenders'])}

        Network Performance:
        - Download Speed: {float(row['rx_avg_bps'])/1_000_000:.2f} Mbps
        - Upload Speed: {float(row['tx_avg_bps'])/1_000_000:.2f} Mbps
        - Signal Strength: {float(row['rssi_mean']):.1f} dBm

        Location: {row['city']}, {row['state']}
        Services: {', '.join([k for k, v in row[['whole_home_wifi', 'wifi_security', 'premium_tech_pro']].items() if v])}
        """

def main():
    try:
        # Initialize processor
        processor = ISPDataProcessor(
            samba_nova_key="your-key",
            csv_path="/content/current_customers-1(Powered by MaxAI).csv"
        )

        # Process embeddings
        vectors = processor.process_and_store_embeddings(batch_size=100)

        print(f"\nProcessed {len(vectors)} customer embeddings")

        # Show sample
        if vectors:
            print("\nSample vector metadata:")
            print(json.dumps(vectors[0]['metadata'], indent=2))

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import pickle

class MemoryEfficientProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def process_embeddings(self, chunk_size: int = 1000) -> None:
        """Process data in chunks to save memory"""
        # Define dtypes to reduce memory usage
        dtypes = {
            'acct_id': str,
            'wireless_clients_count': 'Int32',
            'wired_clients_count': 'Int32',
            'extenders': 'Int32',
            'rx_avg_bps': 'float32',
            'tx_avg_bps': 'float32',
            'rssi_mean': 'float32',
            'city': 'category',
            'state': 'category',
            'whole_home_wifi': bool,
            'wifi_security': bool,
            'premium_tech_pro': bool
        }

        # Initialize chunks iterator
        chunks = pd.read_csv(
            self.csv_path,
            dtype=dtypes,
            chunksize=chunk_size,
            na_values=['NaN', 'null', '', 'NA'],
            low_memory=True
        )

        vectors = []
        total_processed = 0

        # Process each chunk
        for chunk_num, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {chunk_num + 1}")

            # Clean chunk data
            chunk = self._clean_chunk(chunk)

            # Process chunk
            chunk_vectors = self._process_chunk(chunk)

            # Save chunk vectors
            chunk_file = f'embeddings_chunk_{chunk_num}.pkl'
            with open(chunk_file, 'wb') as f:
                pickle.dump(chunk_vectors, f)

            total_processed += len(chunk)
            print(f"Processed {total_processed} records total")

            # Clear memory
            del chunk_vectors

        print(f"\nCompleted processing {total_processed} records")

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean chunk data efficiently"""
        # Fill NA values
        chunk['wireless_clients_count'] = chunk['wireless_clients_count'].fillna(0)
        chunk['wired_clients_count'] = chunk['wired_clients_count'].fillna(0)
        chunk['extenders'] = chunk['extenders'].fillna(0)

        # Fill network metrics with 0 instead of median to save memory
        numeric_cols = ['rx_avg_bps', 'tx_avg_bps', 'rssi_mean']
        chunk[numeric_cols] = chunk[numeric_cols].fillna(0)

        # Fill categorical with 'Unknown'
        chunk['city'] = chunk['city'].fillna('Unknown')
        chunk['state'] = chunk['state'].fillna('Unknown')

        # Fill boolean columns with False
        bool_cols = ['whole_home_wifi', 'wifi_security', 'premium_tech_pro']
        chunk[bool_cols] = chunk[bool_cols].fillna(False)

        return chunk

    def _process_chunk(self, chunk: pd.DataFrame) -> list:
        """Process a single chunk of data"""
        vectors = []

        # Process in smaller batches to manage memory
        batch_size = 100
        for start_idx in tqdm(range(0, len(chunk), batch_size)):
            end_idx = start_idx + batch_size
            batch = chunk.iloc[start_idx:end_idx]

            batch_texts = []
            batch_metadata = []

            for _, row in batch.iterrows():
                # Create text representation
                text = f"""
                Customer {row['acct_id']}:
                Devices: {int(row['wireless_clients_count'])} wireless, {int(row['wired_clients_count'])} wired
                Extenders: {int(row['extenders'])}
                Network: {float(row['rx_avg_bps'])/1e6:.1f} Mbps down, {float(row['tx_avg_bps'])/1e6:.1f} Mbps up
                Signal: {float(row['rssi_mean']):.1f} dBm
                Location: {row['city']}, {row['state']}
                """
                batch_texts.append(text)

                # Create metadata
                metadata = {
                    'devices': {
                        'wireless': int(row['wireless_clients_count']),
                        'wired': int(row['wired_clients_count'])
                    },
                    'network': {
                        'download_mbps': float(row['rx_avg_bps'])/1e6,
                        'upload_mbps': float(row['tx_avg_bps'])/1e6
                    },
                    'location': {
                        'city': str(row['city']),
                        'state': str(row['state'])
                    }
                }
                batch_metadata.append(metadata)

            # Generate embeddings for batch
            embeddings = self.embedding_model.encode(batch_texts)

            # Create vectors
            for idx, embedding in enumerate(embeddings):
                vectors.append({
                    'id': batch.iloc[idx]['acct_id'],
                    'vector': embedding.tolist(),
                    'metadata': batch_metadata[idx]
                })

            # Clear batch data
            del batch_texts
            del batch_metadata
            del embeddings

        return vectors

def main():
    # Initialize processor
    processor = MemoryEfficientProcessor("current_customers.csv")

    # Process data
    processor.process_embeddings(chunk_size=5000)

    # Combine all chunk files (optional)
    print("\nCombining chunk files...")
    all_vectors = []
    import glob
    for chunk_file in glob.glob('embeddings_chunk_*.pkl'):
        with open(chunk_file, 'rb') as f:
            vectors = pickle.load(f)
            all_vectors.extend(vectors)

    # Save combined results
    with open('embeddings_combined.pkl', 'wb') as f:
        pickle.dump(all_vectors, f)

    print(f"Completed! Total vectors: {len(all_vectors)}")

if __name__ == "__main__":
    main()

# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict
import torch
from tqdm import tqdm

class ISPSystem:
    def __init__(self, csv_path: str):
        """Initialize system with data file"""
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.csv_path = csv_path
        self.vectors = None

        # Load and process data
        self._load_and_process_data()

    def _load_and_process_data(self):
        """Load data and create embeddings"""
        print("Loading CSV data...")
        df = pd.read_csv(self.csv_path, dtype={
            'acct_id': str,
            'wireless_clients_count': 'Int32',
            'wired_clients_count': 'Int32',
            'extenders': 'Int32',
            'rx_avg_bps': 'float32',
            'tx_avg_bps': 'float32',
            'rssi_mean': 'float32'
        }).fillna({
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

        # Create embedding matrix for efficient search
        self.embeddings_matrix = np.array([v['vector'] for v in self.vectors])

    def answer_query(self, query: str, top_k: int = 5) -> str:
        """Answer user query"""
        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Find similar profiles
        similarities = np.dot(self.embeddings_matrix, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Generate response based on query type
        if "recommend" in query.lower() or "upgrade" in query.lower():
            return self._generate_recommendations(top_indices)
        elif "issues" in query.lower() or "problems" in query.lower():
            return self._analyze_issues(top_indices)
        else:
            return self._generate_analysis(top_indices)

    def _generate_recommendations(self, indices: List[int]) -> str:
        """Generate recommendations based on similar profiles"""
        profiles = [self.vectors[i]['metadata'] for i in indices]

        response = "Based on similar customer profiles:\n\n"

        # Check for WiFi coverage needs
        high_device_count = sum(1 for p in profiles
                              if p['device_info']['total_devices'] > 8)
        poor_signal = sum(1 for p in profiles
                         if p['network_performance']['signal_strength'] < -70)

        if high_device_count > len(indices)/2:
            response += "1. Recommend Whole Home WiFi due to high device count\n"
        if poor_signal > len(indices)/2:
            response += "2. WiFi extenders would improve coverage\n"

        # Add specific stats
        response += f"\nTypical characteristics of similar customers:"
        response += f"\n- Average devices: {np.mean([p['device_info']['total_devices'] for p in profiles]):.1f}"
        response += f"\n- Average download speed: {np.mean([p['network_performance']['download_mbps'] for p in profiles]):.1f} Mbps"

        return response

    def _analyze_issues(self, indices: List[int]) -> str:
        """Analyze network issues"""
        profiles = [self.vectors[i]['metadata'] for i in indices]

        issues = []
        for profile in profiles:
            if profile['network_performance']['signal_strength'] < -70:
                issues.append("Poor WiFi signal strength")
            if profile['device_info']['total_devices'] > 10 and profile['device_info']['extenders'] == 0:
                issues.append("High device count without extenders")
            if profile['network_performance']['download_mbps'] < 50:
                issues.append("Low download speed")

        if not issues:
            return "No significant issues found in similar profiles."

        return "Common issues found:\n" + "\n".join(f"- {issue}" for issue in set(issues))

    def _generate_analysis(self, indices: List[int]) -> str:
        """Generate general analysis"""
        profiles = [self.vectors[i]['metadata'] for i in indices]

        return f"""Analysis of similar customer profiles:

Device Usage:
- Average wireless devices: {np.mean([p['device_info']['wireless_count'] for p in profiles]):.1f}
- Average wired devices: {np.mean([p['device_info']['wired_count'] for p in profiles]):.1f}

Network Performance:
- Average download: {np.mean([p['network_performance']['download_mbps'] for p in profiles]):.1f} Mbps
- Average signal strength: {np.mean([p['network_performance']['signal_strength'] for p in profiles]):.1f} dBm

Service Adoption:
- Whole Home WiFi: {sum(p['services']['whole_home_wifi'] for p in profiles)/len(profiles)*100:.0f}%
- WiFi Security: {sum(p['services']['wifi_security'] for p in profiles)/len(profiles)*100:.0f}%
"""

def main():
    # Initialize system with the new filename
    system = ISPSystem("current_customers-1(Powered by MaxAI).csv")

    # Example queries
    queries = [
        "What upgrades would you recommend for a customer with 10 devices?",
        "What are common issues for customers with poor signal strength?",
        "What's the typical device usage pattern?",
        "Should I recommend Whole Home WiFi to this customer?"
    ]

    print("\nTesting queries:")
    for query in queries:
        print(f"\nQuery: {query}")
        response = system.answer_query(query)
        print(f"Response:\n{response}")

if __name__ == "__main__":
    main()

# %%
# claude
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm

class ISPRecommendationSystem:
    def __init__(self, csv_path: str):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.csv_path = csv_path
        self.vectors = None

        # Define product catalog
        self.products = {
            'fiber_500': {
                'name': 'Fiber 500',
                'speed': 500,
                'price': 45.00,
                'features': ['Standard WiFi router'],
                'base_product': True
            },
            'fiber_1gig': {
                'name': 'Fiber 1 Gig',
                'speed': 1000,
                'price': 65.00,
                'features': ['Standard WiFi router'],
                'base_product': True
            },
            'fiber_2gig': {
                'name': 'Fiber 2 Gig',
                'speed': 2000,
                'price': 99.00,
                'features': ['Upgraded WiFi router', 'One extender included'],
                'base_product': True
            },
            'wifi_security': {
                'name': 'WiFi Security',
                'price': 5.00,
                'features': ['Malicious site protection', 'Parental controls', 'Ad blocking'],
                'addon': True
            },
            'whole_home_wifi': {
                'name': 'Whole Home WiFi',
                'price': 10.00,
                'features': ['Up to two extenders', 'Strong WiFi throughout home'],
                'addon': True
            }
        }

        # Load customer data
        self.load_and_process_data()  # Changed from _load_and_process_data

    def load_and_process_data(self):  # Changed method name to match the call
        """Load data and create embeddings"""
        print("Loading CSV data...")
        df = pd.read_csv(self.csv_path, dtype={
            'acct_id': str,
            'wireless_clients_count': 'Int32',
            'wired_clients_count': 'Int32',
            'extenders': 'Int32',
            'rx_avg_bps': 'float32',
            'tx_avg_bps': 'float32',
            'rssi_mean': 'float32'
        }).fillna({
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

# Rest of the code remains the same...

def main():
    # Initialize system
    system = ISPRecommendationSystem("current_customers-1(Powered by MaxAI).csv")

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
    main()


# %%
# claude
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm

class ISPRecommendationSystem:
    def __init__(self, csv_path: str):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.csv_path = csv_path
        self.vectors = None

        # Define product catalog
        self.products = {
            'fiber_500': {
                'name': 'Fiber 500',
                'speed': 500,
                'price': 45.00,
                'features': ['Standard WiFi router'],
                'base_product': True
            },
            'fiber_1gig': {
                'name': 'Fiber 1 Gig',
                'speed': 1000,
                'price': 65.00,
                'features': ['Standard WiFi router'],
                'base_product': True
            },
            'fiber_2gig': {
                'name': 'Fiber 2 Gig',
                'speed': 2000,
                'price': 99.00,
                'features': ['Upgraded WiFi router', 'One extender included'],
                'base_product': True
            },
            'wifi_security': {
                'name': 'WiFi Security',
                'price': 5.00,
                'features': ['Malicious site protection', 'Parental controls', 'Ad blocking'],
                'addon': True
            },
            'whole_home_wifi': {
                'name': 'Whole Home WiFi',
                'price': 10.00,
                'features': ['Up to two extenders', 'Strong WiFi throughout home'],
                'addon': True
            }
        }

        # Load customer data
        self.load_and_process_data()  # Changed from _load_and_process_data

    def load_and_process_data(self):  # Changed method name to match the call
        """Load data and create embeddings"""
        print("Loading CSV data...")
        df = pd.read_csv(self.csv_path, dtype={
            'acct_id': str,
            'wireless_clients_count': 'Int32',
            'wired_clients_count': 'Int32',
            'extenders': 'Int32',
            'rx_avg_bps': 'float32',
            'tx_avg_bps': 'float32',
            'rssi_mean': 'float32'
        }).fillna({
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
        """Analyze a scenario and provide a recommendation"""
        # For simplicity, return a placeholder recommendation
        return "Based on the scenario, we recommend the 'Fiber 500' plan with 'WiFi Security' as an add-on."

def main():
    # Initialize system
    system = ISPRecommendationSystem("current_customers-1(Powered by MaxAI).csv")

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
    main()


# %%

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict
from tqdm import tqdm

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

if __name__ == "__main__":
    main()


# %%

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict
from tqdm import tqdm
import time  # Added time import

def test_scenarios():
    # Initialize system
    system = ISPRecommendationSystem(
        csv_path="current_customers-1(Powered by MaxAI).csv",
        samba_nova_key="ea5bee1c-32d8-4302-9d51-fed6523192c7"
    )

    scenarios = [
        # Budget-conscious scenarios
        """
        Single student living in a studio apartment:
        - Limited budget (maximum $50/month)
        - Uses internet for online classes and streaming
        - Has a laptop and smartphone
        - Basic security needs
        """,

        # Family scenarios
        """
        Large family of 6:
        - Parents working from home
        - 4 kids doing online schooling
        - Multiple smart TVs and gaming consoles
        - Security and parental controls are crucial
        - Large 3-bedroom house with basement
        - Budget up to $150/month
        """,

        # Professional scenarios
        """
        Software developer working remotely:
        - Needs very high-speed internet
        - Large file uploads/downloads
        - Multiple development servers
        - Video conferencing
        - Lives in a high-rise apartment
        - Budget isn't a constraint
        """
    ]  # Reduced number of scenarios for testing

    print("Starting scenario testing...\n")

    # Test each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {i}:")
        print(scenario.strip())
        print("\nAnalyzing scenario...")

        try:
            response = system.analyze_scenario(scenario)
            print("\nRecommendations:")
            print(response)
        except Exception as e:
            print(f"Error processing scenario {i}: {str(e)}")

        print(f"{'='*80}")

        # Add delay between scenarios
        if i < len(scenarios):  # Don't sleep after last scenario
            print("\nProcessing next scenario in 2 seconds...")
            time.sleep(2)

def main():
    print("Testing ISP Recommendation System...")
    test_scenarios()
    print("\nTesting completed!")

if __name__ == "__main__":
    main()

# %%
class ScenarioGenerator:
    def __init__(self):
        # Define components for scenario building
        self.usage_profiles = {
            "streaming": {
                "basic": "Occasional Netflix and YouTube",
                "moderate": "Daily streaming on multiple devices",
                "heavy": "4K streaming and content creation",
                "professional": "Multiple 4K streams and content production"
            },
            "gaming": {
                "casual": "Occasional online gaming",
                "regular": "Daily gaming sessions",
                "competitive": "Competitive gaming and streaming",
                "professional": "Esports and professional streaming"
            },
            "work": {
                "basic": "Email and web browsing",
                "moderate": "Regular video calls",
                "heavy": "Multiple video conferences",
                "professional": "Development and large file transfers"
            }
        }

        self.device_profiles = {
            "minimal": ["Smartphone", "Laptop"],
            "standard": ["Smartphones", "Laptops", "Smart TV"],
            "advanced": ["Multiple smartphones", "Gaming PC", "Smart TVs", "Tablets"],
            "professional": ["Workstations", "Servers", "Multiple monitors", "Professional equipment"]
        }

        self.location_types = {
            "small": "Studio apartment",
            "medium": "2-bedroom apartment",
            "large": "3-bedroom house",
            "complex": "Multi-story house with basement"
        }

        self.budget_ranges = {
            "budget": "Under $50/month",
            "standard": "$50-100/month",
            "premium": "$100-200/month",
            "unlimited": "No budget constraints"
        }

    def generate_scenario(self,
                         usage_type: str,
                         intensity: str,
                         devices: str,
                         location: str,
                         budget: str,
                         additional_reqs: List[str] = None) -> str:
        """Generate a detailed scenario"""
        scenario = f"""
        Customer Profile:
        - Usage: {self.usage_profiles[usage_type][intensity]}
        - Devices: {', '.join(self.device_profiles[devices])}
        - Location: {self.location_types[location]}
        - Budget: {self.budget_ranges[budget]}
        """

        if additional_reqs:
            scenario += "\nAdditional Requirements:\n"
            scenario += "\n".join(f"- {req}" for req in additional_reqs)

        return scenario.strip()

def test_comprehensive_scenarios():
    # Initialize system and scenario generator
    system = ISPRecommendationSystem(
        csv_path="current_customers-1(Powered by MaxAI).csv",
        samba_nova_key="ea5bee1c-32d8-4302-9d51-fed6523192c7"
    )
    generator = ScenarioGenerator()

    # Define test scenarios
    scenarios = [
        # Ultra-high performance scenario
        generator.generate_scenario(
            usage_type="gaming",
            intensity="professional",
            devices="professional",
            location="large",
            budget="unlimited",
            additional_reqs=[
                "Need lowest possible latency",
                "24/7 uptime requirement",
                "Multiple simultaneous 4K streams",
                "Professional tech support",
                "Backup internet connection"
            ]
        ),

        # Mixed use family scenario
        generator.generate_scenario(
            usage_type="streaming",
            intensity="heavy",
            devices="advanced",
            location="complex",
            budget="premium",
            additional_reqs=[
                "Parental controls required",
                "Whole-home WiFi coverage",
                "Security features needed",
                "Smart home integration"
            ]
        ),

        # Small business scenario
        generator.generate_scenario(
            usage_type="work",
            intensity="professional",
            devices="advanced",
            location="medium",
            budget="premium",
            additional_reqs=[
                "Business-critical reliability",
                "VPN support",
                "Static IP address",
                "Cloud backup capabilities"
            ]
        ),

        # Budget student scenario
        generator.generate_scenario(
            usage_type="streaming",
            intensity="moderate",
            devices="minimal",
            location="small",
            budget="budget",
            additional_reqs=[
                "Basic security features",
                "Suitable for video calls",
                "Monthly data usage < 500GB"
            ]
        ),

        # Smart home enthusiast
        generator.generate_scenario(
            usage_type="streaming",
            intensity="heavy",
            devices="advanced",
            location="large",
            budget="premium",
            additional_reqs=[
                "IoT device support",
                "Advanced security features",
                "Mesh network capability",
                "Home automation support"
            ]
        ),

        # Content creator setup
        generator.generate_scenario(
            usage_type="streaming",
            intensity="professional",
            devices="professional",
            location="medium",
            budget="premium",
            additional_reqs=[
                "High upload speeds",
                "Stream quality priority",
                "Equipment optimization",
                "Content delivery optimization"
            ]
        ),

        # Hybrid work professional
        generator.generate_scenario(
            usage_type="work",
            intensity="heavy",
            devices="advanced",
            location="medium",
            budget="standard",
            additional_reqs=[
                "Video conferencing quality",
                "File sharing capability",
                "Work hours reliability",
                "Basic security features"
            ]
        ),

        # Multi-generational family
        generator.generate_scenario(
            usage_type="streaming",
            intensity="heavy",
            devices="advanced",
            location="complex",
            budget="premium",
            additional_reqs=[
                "Easy-to-use interface",
                "Multiple user profiles",
                "Strong WiFi coverage",
                "Flexible parental controls"
            ]
        ),

        # Competitive gaming team
        generator.generate_scenario(
            usage_type="gaming",
            intensity="professional",
            devices="professional",
            location="large",
            budget="unlimited",
            additional_reqs=[
                "Ultra-low latency",
                "Quality of Service (QoS)",
                "Traffic prioritization",
                "Performance monitoring"
            ]
        ),

        # Home security focus
        generator.generate_scenario(
            usage_type="streaming",
            intensity="heavy",
            devices="advanced",
            location="large",
            budget="premium",
            additional_reqs=[
                "Camera system support",
                "Continuous monitoring",
                "Backup connection",
                "Security features"
            ]
        )
    ]

    print("Starting comprehensive scenario testing...\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {i}:")
        print(scenario)
        print("\nAnalyzing scenario...")

        try:
            response = system.analyze_scenario(scenario)
            print("\nRecommendations:")
            print(response)
        except Exception as e:
            print(f"Error processing scenario {i}: {str(e)}")

        print(f"{'='*80}")

        if i < len(scenarios):
            print("\nProcessing next scenario in 2 seconds...")
            time.sleep(2)

def main():
    print("Testing Comprehensive ISP Scenarios...")
    test_comprehensive_scenarios()
    print("\nTesting completed!")

if __name__ == "__main__":
    main()


# %%



