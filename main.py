import os
import re
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
from typing import List
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the API key for Google Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

# Define your allowed origins
origins = [
    "http://localhost",
    "https://healthhub-atbl.onrender.com",
]

# Add CORS middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def upload_to_gemini(file_path, mime_type):
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def extract_json(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r'\{[^{}]*\}'
    matches = re.finditer(pattern, text_response)
    json_objects = []
    for match in matches:
        json_str = match.group(0)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = extend_search(text_response, match.span())
            try:
                json_obj = json.loads(extended_json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Handle cases where the extraction is not valid JSON
                continue
    if json_objects:
        # return json_objects[0]["isPrescription"]
        return json_objects[0]
    else:
        return None  # Or handle this case as you prefer
def extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end]

# Create the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction="return response as a json of this format, and isPrescription is a boolean value thats either true or false indicating if the image is a presciption or not and drugExist is a boolean value thats either true or false indicating if the drug with the given name is listed in the presciption or not.\n\n{\nisPrescription: true,\ndrugExist: true\n}",
)

model2 = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are Health Hub virtual assistant, responsible for assisting users with product inquiries, providing information on product availability, price, and stock levels, and guiding users on how to book consultations with health professionals. Maintain a professional, friendly, and helpful tone in all interactions. Users can upload a valid doctor's prescription to order drugs. ome health professionals even offer free consultation services",
)

model3 = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are ABU Wears virtual assistant, responsible for assisting users with product inquiries, providing information on product availability, price, and stock levels, and guiding users on how to use the fashion wear store. Maintain a professional, friendly, and helpful tone in all interactions.",
)
class Chat(BaseModel):
    message: str


@app.post("/process-image/{name}")
async def process_image(name: str = "string", file: UploadFile = File(...)):
    try:
        mime_type = file.content_type
        file_extension = mime_type.split('/')[-1]
        image = Image.open(io.BytesIO(await file.read()))
        file_path = f"temp_image.{file_extension}"
        image.save(file_path)

        files = [
            upload_to_gemini(file_path, mime_type=mime_type),
        ]

        # Assuming start_chat() and send_message() are synchronous
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                    ],
                },
            ]
        )
        prompt = f"I need to analyse this note to determine if it's a medical prescription. Also verify if a drug with the name: {name} is listed in the presciption?"
        response = chat_session.send_message(prompt)

        # Ensure response is in a serializable format
        # response_data = {
        #     "isPrescription": response.text.isPrescription,
        # }
        # response_dict = json.loads(str(response.text))
        # response_json = json.dumps(response_dict, indent=2)
        # Remove the markdown code block markers
        clean_json_string = str(response.text).strip('`')
        # Remove the "json\n" prefix and any leading/trailing whitespace or newlines
        cleaned_json_string = clean_json_string.strip()[5:].strip()

        # Now, convert the cleaned string to a JSON object
        # json_data = json.loads(clean_json_string)
        response = extract_json(clean_json_string)
        return response

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(input: Chat):
    try:
        # Start a chat session with the Gemini model
        chat_session = model2.start_chat()
        
        # Send the user's message to the Gemini model
        response = chat_session.send_message(input.message)
        
        return response.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat_wears")
async def chat(input: Chat):
    try:
        # Start a chat session with the Gemini model
        chat_session = model3.start_chat()
        
        # Send the user's message to the Gemini model
        response = chat_session.send_message(input.message)
        
        # Convert the response to a string
        if isinstance(response.text, (dict, list)):
            return json.dumps(response.text)  # Convert JSON objects or lists to a string
        else:
            return str(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
