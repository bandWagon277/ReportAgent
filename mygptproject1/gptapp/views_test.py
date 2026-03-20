import os
import requests

refine_prompt = "For the following request of producing a csv analysis script，please generate as the order 1.example csv；2.sas code；3.python code. No explanation needed.Script can be executed with any given csv file path. Request:"
user_prompt = "Please count the number of rows"
# Call the OpenAI GPT API with the user-provided prompt
payload = {
            "messages": [
                {"role": "user", "content": refine_prompt + user_prompt}
            ],
            #"model": "gpt-3.5-turbo",
            "model": "gpt-4o",
        }
api_key = os.getenv('OPENAI_API_KEY', '')
headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
content = response.json()
generated_text = content.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
print(generated_text)
