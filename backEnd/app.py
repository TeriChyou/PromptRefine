from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import openai

# Import custom modules
import dataAnalyze

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Functions
def read_file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content

@app.route('/', methods=['GET'])
def hello():
    return "Hello World"

# Configure the OpenAI GPT model
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("Error: OPENAI_API_KEY environment variable is not set!")
conversation_history = []

# Load system instruction from the file
file_path = os.path.abspath(r'backEnd/promptList/prompt.txt')
system_instruction = read_file_to_string(file_path)

# Basic GPT generate function
@app.route('/python-api/gpt-generate', methods=['POST'])
def generate_gpt_response():
    data = request.json
    question = data.get('question', 'No question provided.')
    answer = data.get('answer', 'No answer provided.')
    # 手動key的次要PROMPT和正確答案及關鍵字 (20241017)
    prompt = data.get('prompt', 'No second prompt provided.')
    # 新增正確答案、關鍵字 (20241021)
    correctAnswer = data.get('correctAnswer', 'No correct answer provided.')
    keyWords = data.get('keyWords', 'No key words provided.')

    try:
        # 將使用者的回答加入對話紀錄 
        input_text = f"Question: {question}\n Student's Answer: {answer if answer.strip() else 'No valid answer provided'}\n"
        conversation_history.append({"role":"user", "content":input_text})
        
        # Call OpenAI's GPT-4o API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages = [
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "System instructions:" + system_instruction # + "SECOND PROMPT:" + prompt + "Reference Correct Answer:" + correctAnswer + "Key words:" + keyWords
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": input_text
                    }
                ]
                }
            ] ,
            temperature=1, # temperature => AI回饋的多樣性，2是最高，0是最低
            max_tokens=16384, # 基本上要翻Document看call哪支的 來確認maximum token
            top_p=0.95, # top_p => AI抽樣機率分布的範圍，1是最高，0是最低
            frequency_penalty=0, # frequency_penalty => -2是最低，2是最高 越高在單句的回覆內越不會重複同樣字眼
            presence_penalty=0 # presence_penalty => -2是最低，2是最高 越高在複數的回應內越不會重複同樣字眼 (對於有歷史對話影響較多)
        )
        
        # Extract the response text from GPT-4
        gpt_response_text = response.choices[0].message.content.strip()

        # Parse the response text as JSON
        gpt_response_json = json.loads(gpt_response_text)

        return jsonify(gpt_response_json)
    except json.JSONDecodeError:
        # If parsing fails, return an error message
        return jsonify({"error": "Invalid JSON response from GPT", "text": gpt_response_text}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Analyze application PDF with GPT
@app.route('/python-api/analyzeApplication', methods=['POST'])
def generate_application_feedback():
    data = request.json
    file_path = data.get('file_path', 'No file path provided.')
    question = data.get('question', 'No question provided.')
    answer = data.get('answer', 'No answer provided.')

    try:
        # Read the PDF file and extract text (assuming the function is defined elsewhere)
        text = dataAnalyze.extract_file_content(file_path)

        # Call OpenAI's GPT-4o API with the extracted text and question
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_instruction + f"Question: {question}\nAnswer: {answer}\n\n{file_path}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=16384,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract the response text from GPT-4
        gpt_response_text = response.choices[0].message.content.strip()

        # Parse the response text as JSON
        gpt_response_json = json.loads(gpt_response_text)

        return jsonify(gpt_response_json)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from GPT"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app on localhost
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
