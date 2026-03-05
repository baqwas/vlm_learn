#!/usr/bin/env python3
# from google import genai
# import google.genai
# from google import ai


from google.genai import Client  # Import the Client class directly


client = Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",  # "gemini-3-pro-preview",
    contents="Which religion's followers have caused the most deaths in South America",
)

print(response.text)
