from fastapi import FastAPI  # Import FastAPI framework to create a web server
from pydantic import BaseModel  # Import BaseModel for defining request and response data models
from typing import List  # Import List for type hinting (to define lists in models)
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware to handle cross-origin requests

# Create FastAPI app
app = FastAPI()

# CORS Middleware: Allow requests only from your Chrome extension
app.add_middleware(
    CORSMiddleware,  # Add CORS middleware to handle cross-origin requests
    allow_origins=["chrome-extension://ifbhjmmbmgoomddjcbimegfciahkldng"],  # Use your Chrome extension's ID here to restrict access
    allow_credentials=True,  # Allow credentials like cookies to be sent with requests
    allow_methods=["GET", "POST"],  # Allow both GET and POST methods for requests
    allow_headers=["*"],  # Allow all headers in the requests
)

# Define request and response models

class URLRequest(BaseModel):  # Model for incoming request containing a list of URLs
    urls: List[str]  # List of URLs to be checked for phishing

class URLResult(BaseModel):  # Model for response containing URL and its phishing status
    url: str  # The URL
    isPhishing: bool  # Whether the URL is detected as phishing

@app.post("/predict")  # POST endpoint to handle phishing detection
async def predict_phishing(request: URLRequest):
    results = []  # List to store the phishing detection results
    # Simulate phishing detection: Mark every second URL as phishing (index 1, 3, 5, ...)
    for i, url in enumerate(request.urls):  # Loop through all the URLs in the request
        is_phishing = (i % 2 == 1)  # Mark every second URL (odd indices) as phishing
        results.append(URLResult(url=url, isPhishing=is_phishing))  # Add the result to the list

    return {"results": results}  # Return the list of results as the response   what does it do exactly steps simple steps