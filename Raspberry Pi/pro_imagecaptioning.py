###pip install google-generativeai#####

import cv2
import google.generativeai as genai
from pathlib import Path
import pyttsx3
import threading
import os

def say(text):
    engine = pyttsx3.init()
    engine.setProperty('sapi', 10)
    engine.say(text)
    engine.runAndWait()

# Initialize webcam
cap = cv2.VideoCapture(0)
genai.configure(api_key='AIzaSyB-IbJ3LvK2x10tWXxcDcQv_2CRoYBLPQI')

# Initialize counter
counter = 0

generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


def input_image_setup(file_loc):
    if not (img := Path(file_loc)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path(file_loc).read_bytes()
        }
    ]
    return image_parts


def generate_gemini_response_async(input_prompt, image_loc, question_prompt):
    response = generate_gemini_response(input_prompt, image_loc, question_prompt)
    print(response)


def generate_gemini_response(input_prompt, image_loc, question_prompt):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return response.text


def delete_saved_images():
    for i in range(1, counter + 1):
        filename = f'Videos/image_{i}.png'
        if os.path.exists(filename):
            os.remove(filename)


input_prompt = """"
               You are an expert in understanding scenarios and reading texts and identifying objects.
               You will receive input images as scenarios &
               you will have to describe scenarios and identify objects and read texts based on the input image
               """

# Register an exit handler to delete the saved images before exiting
import atexit
atexit.register(delete_saved_images)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Check for the 's' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Increment counter
        counter += 1

        # Save the image with counter appended to the filename
        filename = f'Videos/image_{counter}.png'
        cv2.imwrite(filename, frame)
        image_loc = f"Videos/image_{counter}.png"
        question_prompt = "What is this image? describe precisely"

        # Use a separate thread to generate Gemini response asynchronously
        threading.Thread(target=generate_gemini_response_async, args=(input_prompt, image_loc, question_prompt)).start()

        print(f"Image {counter} saved as {filename}")

    # Check for the 'q' key press to exit the loop
    elif key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
