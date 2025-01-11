import os
import sys
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from glm.glm_response import glm_response
from o1.g1 import generate_g1_response
from o1.pysolver import get_pysolver_response

def get_response(prompt, model = "glm-4-flash"):
    if "glm" in model:
        return glm_response(prompt = prompt, model = model)

def get_few_shot_prompt(subject, level, num = 3):
    id = 0
    prompt = ""
    file_subject = ""
    if subject == "Precalculus":   
        file_subject = "precalculus"
    elif subject == "Number Theory":
        file_subject = "number_theory"
    elif subject == "Intermediate Algebra":
        file_subject = "intermediate_algebra"
    elif subject == "Prealgebra":
        file_subject = "prealgebra"
    elif subject == "Algebra":
        file_subject = "algebra"

    for i in range(30000):
        data_dir = os.path.join(parent_dir, f"data/math/train/{file_subject}/")
        data_filepath = os.path.join(data_dir, f"{i}.json")
        if os.path.exists(data_filepath):
            with open(data_filepath, "r") as f:
                data = json.load(f)
                if data["level"] == "Level " + str(level):
                    prompt += f"Question: {data['problem']}\nSolution: {data['solution']}\n\n"
                    id += 1
        if id == num:
            break

    return prompt

def get_answer(question, subject, level, method = "default", model = "glm-4-flash"):
    if method == "default":
        prompt = f"Question: {question}\nSubject: {subject}\nLevel: {level}\nAnswer:"
        return get_response(prompt, model)
    elif method == "3shot":
        prompt = get_few_shot_prompt(subject, level) + f"Question: {question}\nSubject: {subject}\nLevel: {level}\nAnswer:"
        # print(prompt)
        return get_response(prompt, model)
    elif method == "5shot":
        prompt = get_few_shot_prompt(subject, level, 5) + f"Question: {question}\nSubject: {subject}\nLevel: {level}\nAnswer:"
        # print(prompt)
        return get_response(prompt, model)
    elif method == "g1":
        steps, total_time, final_answer = generate_g1_response(question, model)
        return final_answer
    elif method == "py_solver":
        code, final_answer = get_pysolver_response(question, model)
        return final_answer
    else:
        return "Method not implemented"
    
if __name__ == '__main__':
    print(get_response("1+1="))
    print(get_answer("1+1=", "math", 3))
    print(get_answer("1+1=", "Number Theory", 3, "3shot"))