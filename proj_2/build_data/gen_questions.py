from openai import OpenAI
import json
import os

client = OpenAI(api_key="sk-0e7ceede97ed4264bb24349284f57ce6", base_url="https://api.deepseek.com")
question_prefix = """
Here is a text. Please generate 50 questions based on the information in the text. The questions should be answerable using the information from the text. Only output the questions.

EXAMPLE JSON OUTPUT:
{
   "questions":[
      "the first question",
      "the second question",
      "the third question"
   ]
}

"""

def generate_question(client, text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question_prefix + text},
        ],
        response_format={
            'type': 'json_object'
        },
        stream=False,
        temperature=0.8,
        max_tokens=8192
    )
    
    return response.choices[0].message.content

if __name__ == '__main__':
    f1 = open("data/questions.json", "r")
    c1 = json.load(f1)
    f1.close()

    text = open("data/book.txt", "r").read()
    q = generate_question(client=client, text=text)
    open("data/questions2.json", "w").write(q)

    f2 = open("data/questions2.json", "r")
    c2 = json.load(f2)
    f2.close()

    os.remove("data/questions2.json")

    merged_json = {
        "questions": c1["questions"] + c2["questions"]
    }
    print(f"Current questions count={len(merged_json["questions"])}")
    f = open("data/questions.json", "w")
    json.dump(merged_json, f)
    f.close()


