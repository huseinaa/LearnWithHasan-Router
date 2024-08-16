import json
import time
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.tools.generic_loader import load_content

def fetch_apis(filepath):
    text_file = load_content(filepath)
    content = json.loads(text_file.content)
    return content
  
def find_best_api(user_query, api_descriptions):
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini")
    u_prompt = f"""
    You are an expert in problem solving. I have a user has a specific query and I want to check if I have an API
    that would help him solve this problem. I'll give you both the user inquery and the list of APIs in the inputs
    section delimited between triple backticks. So analyze both of them very well and check if there's an API which
    can help him or no. 

    ##Inputs
    user inquiry: ```[{user_query}]```
    API list: ```[{api_descriptions}]```

    #Output
    The output should only be the API name as provided in the inputs and nothing else. If no API was found return None.
    """

    response = llm_instance.generate_response(prompt=u_prompt)
    return response

filepath = 'apis.json' 
api_descriptions = fetch_apis(filepath)

user_query = input("Enter your inquiry: ")

start_time = time.time() # Start timer

result = find_best_api(user_query, api_descriptions)
    
if result!="None":
    print(f"The best API to use is: {result}")
else:
    print("No suitable API found.")

end_time = time.time()  # End timer
print(f"Execution time: {end_time - start_time} seconds")
