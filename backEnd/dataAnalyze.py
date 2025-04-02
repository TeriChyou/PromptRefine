import os
from PyPDF2 import PdfReader
import openai
# custom modules
import functions as func


# 處理檔案
def extract_file_content(file_path):
    extension = os.path.splitext(file_path)[-1].lower()

    try:
        if extension == '.pdf':
            reader = PdfReader(file_path)
            text = ''.join([page.extract_text() for page in reader.pages])
            return text

        else:
            return f"Unsupported file type: {extension}"

    except Exception as e:
        return f"Error processing file: {e}"

system_instructions = func.read_file_to_string(r'backEnd\promptList\applicationInstruciton.txt')
# with files
def analyze_with_gpt(file_content, instruction):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": f"{instruction}\nHere is the file content:\n{file_content}"}
            ],
            temperature=0.5, # temperature => AI回饋的多樣性，2是最高，0是最低
            max_tokens=16384, # 基本上要翻Document看call哪支的 來確認maximum token
            top_p=0.95, # top_p => AI抽樣機率分布的範圍，1是最高，0是最低
            frequency_penalty=0, # frequency_penalty => -2是最低，2是最高 越高在單句的回覆內越不會重複同樣字眼
            presence_penalty=0 # presence_penalty => -2是最低，2是最高 越高在複數的回應內越不會重複同樣字眼 (對於有歷史對話影響較多)
        )
        func.tokenUsed(response.usage.total_tokens)
        res = response.choices[0].message.content
        return res
    except Exception as e:
        return f"Error during GPT analysis: {e}"
