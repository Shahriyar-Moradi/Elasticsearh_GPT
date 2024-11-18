from dotenv import load_dotenv
load_dotenv() 
import os
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
print(client.api_key)