import os
import re
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
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

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="return response as a json of this format, and isPrescription is a boolean value thats either true or false.\n\n{\nisPrescription: true\n}",
)


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
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
        prompt = "I need to analyse this note to determine if it's a medical prescription or not."
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
        return extract_json(clean_json_string)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
