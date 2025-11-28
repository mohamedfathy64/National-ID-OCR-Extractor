import requests
import base64
import json
import time
import os

# --- Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual API key.
# You can get a key from Google AI Studio.
API_KEY = "PUT_YOUR_API_KEY_HERE" 
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# --- Utility Functions ---

def image_to_base64(filepath):
    """Converts a local image file to a base64 encoded string."""
    print(f"Loading image from: {filepath}")
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at path: {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return None

def extract_id_from_image(image_path):
    """
    Sends the image to the Gemini API for OCR and National ID extraction.
    
    Args:
        image_path (str): The file path to the image containing the ID.

    Returns:
        str: The extracted 14-digit National ID number, or None if extraction fails.
    """
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return None
    
    # We assume the image is a common format like JPEG. Adjust if necessary.
    # The original image uploaded was JPEG, so we keep this default.
    mime_type = "image/jpeg" 

    # --- API Request Setup ---
    # The system instruction tells the model exactly what persona to adopt and what to output.
    system_instruction = (
        "You are an OCR and data extraction expert focused on identifying Egyptian National "
        "ID numbers (14 digits) from images. The number is typically written in Eastern "
        "Arabic numerals. You MUST find the 14-digit sequence, convert it to Western Arabic "
        "numerals (0-9), and return *only* the 14-digit string. Do not include spaces, "
        "explanations, or any surrounding text. The required length is exactly 14 characters."
    )
    
    # The user query is simple, as the hard work is defined in the system instruction.
    user_query = "Extract the 14-digit National ID from the image."

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_query},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
    }

    url = API_URL_TEMPLATE.format(model=MODEL_NAME, key=API_KEY)
    
    # --- API Call with Exponential Backoff ---
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempting API call (Attempt {attempt + 1}/{max_retries})...")
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            
            # Extract the generated text
            if result.get('candidates'):
                text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Simple validation and cleanup to ensure it's 14 digits
                cleaned_id = ''.join(filter(str.isdigit, text))
                
                if len(cleaned_id) == 14:
                    print("Extraction successful.")
                    return cleaned_id
                else:
                    print(f"Warning: Extracted string was not 14 digits after cleanup. Length: {len(cleaned_id)}. Raw output: {text}")
                    # Try to use a manual conversion map as a final fallback
                    return manual_conversion_fallback(text)
            
            print("API response did not contain valid content.")
            return None

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if e.response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds due to API error...")
                time.sleep(delay)
            else:
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    
    print("API call failed after multiple retries.")
    return None

def manual_conversion_fallback(text):
    """
    A robust Python function to manually convert Eastern Arabic digits
    to Western Arabic digits, filtering for 14 digits.
    This serves as a final check if the model returns the wrong format.
    """
    ARABIC_TO_WESTERN = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    
    western_digits = ''.join(ARABIC_TO_WESTERN.get(c, c) for c in text)
    
    # Filter for only digits and take the first 14 characters
    final_id = ''.join(filter(str.isdigit, western_digits))
    
    if len(final_id) >= 14:
        return final_id[:14]
    
    return None

# --- Main Execution ---

if __name__ == "__main__":
    
    if os.environ.get('GEMINI_API_KEY'):
        API_KEY = os.environ.get('GEMINI_API_KEY')

    # This check now prints the error if the key is still the placeholder.
    if API_KEY == "PASTE_YOUR_COPIED_KEY_HERE":
        print("\n*** ERROR: Please replace 'PASTE_YOUR_COPIED_KEY_HERE' in the script with your actual API key. ***\n")
    else:
        # 1. Ask the user for the image path
        IMAGE_FILE_PATH = input("Please enter the full path to the ID image file: ")
        
        print("--- Starting National ID Extraction ---")
        extracted_id = extract_id_from_image(IMAGE_FILE_PATH)

        if extracted_id:
            print("\n=============================================")
            print(f"✅ Extracted National ID (14 digits): {extracted_id}")
            print("=============================================")
        else:
            print("\n❌ Failed to extract the 14-digit National ID.")
            

        print("--- Extraction Complete ---")
