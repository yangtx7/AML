import time
import re
import os
import sys
import json
import subprocess
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llm.glm.glm_response import glm_response
import re

def execute_python_code(code: str, timeout: int = 30):
    open("tmp.py", "w").write(code)
    try:
        # Run the Python subprocess with a timeout
        result = subprocess.run(
            ["python", "tmp.py"],
            text=True,
            capture_output=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "output": f"Timeout: The script took longer than {timeout} seconds to execute."
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"Error: {str(e)}"
        }
    finally:
        if os.path.exists("tmp.py"):
            os.remove("tmp.py")


def extract_python_code(text):
    code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(code_blocks)

def get_pysolver_response(prompt, model="glm-4-flash"):
    try_cnt = 0
    while try_cnt <= 5:
        try_cnt += 1
        message_prefix = "Write a Python program to solve the following math problem. You can call various library functions, including numpy, scipy, panda, torch, etc. The Python program needs to be able to output intermediate results and final results, and the program needs to have comments. Notice: Only output the code.\n"

        messages = [
            {"role": "system", "content": "You are a master at writing Python programs to solve math problems."},
            {"role": "user", "content": message_prefix + prompt}
        ]
        
        response = glm_response(messages=messages, model=model)
        py_code = extract_python_code(response)
        result = execute_python_code(py_code)
        if result["success"]:
            break

    return py_code, result["output"]


if __name__ == '__main__':
    prompt = "how many r in strawberry"
    prompt = "How many positive whole-number divisors does 196 have?"
    py_code, final_answer = get_pysolver_response(prompt)
    print(py_code)
    print("=" * 40)
    print(final_answer)
