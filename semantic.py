import json
import time
import numpy as np
from SimplerVectors_core import VectorDatabase
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider

def fetch_apis(filepath):
    text_file = load_content(filepath)
    content = json.loads(text_file.content)
    return [(api['name'], api['description']) for api in content] 

def get_embeddings(texts):
    try:
        embeddings_instance = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
        response = embeddings_instance.generate_embeddings(texts)
        
        embeddings = np.array([item.embedding for item in response])
        return embeddings
    
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])

def find_best_api(user_query, apis):
    db = VectorDatabase('VDB')

    api_descriptions = [api[1] for api in apis] 
    query_embedding = get_embeddings([user_query])
    description_embeddings = get_embeddings(api_descriptions)

    for idx, emb in enumerate(description_embeddings):
     db.add_vector(emb, {"api": apis[idx][0], "description": apis[idx][1]}, normalize=True)

    query_embedding = db.normalize_vector(query_embedding[0])
    best_sim = db.top_cosine_similarity(query_embedding, top_n=1)

    if best_sim:
        return best_sim[0]
    else:
        return None

# Input:
filepath = 'apis.json' 

apis = fetch_apis(filepath)
user_query = input("Enter your inquiry: ")

start_time = time.time() # Start timer

result = find_best_api(user_query, apis)

if result[1] > 0.3:
    print(f"The best API to use is: {result[0]["api"]}")
    print(f"Info about this result: {result}")
else:
    print("No suitable API found.")

end_time = time.time()  # End timer
print(f"Execution time: {end_time - start_time} seconds")
