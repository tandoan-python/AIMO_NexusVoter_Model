import re
import sys
import time
import math
import traceback
import contextlib
import multiprocessing
from collections import Counter
from io import StringIO

# Kaggle Evaluation API & Polars
import polars as pl
import aimo_3_inference_server

# vLLM
from vllm import LLM, SamplingParams

# ==========================================
# 1. PYTHON REPL (MÔI TRƯỜNG THỰC THI CODE)
# ==========================================
class PythonREPL:
    """
    Môi trường thực thi code Python cô lập, có giới hạn thời gian (Timeout).
    Bảo vệ hệ thống khỏi lỗi lặp vô tận (Runtime Error).
    """
    def __init__(self, timeout=30):
        self.timeout = timeout

    def _execute_in_process(self, code: str, return_dict: dict):
        output_buffer = StringIO()
        with contextlib.redirect_stdout(output_buffer):
            try:
                exec_globals = {'math': math, 'sympy': __import__('sympy')}
                exec(code, exec_globals)
                return_dict['output'] = output_buffer.getvalue()
                return_dict['error'] = None
            except Exception as e:
                return_dict['output'] = output_buffer.getvalue()
                return_dict['error'] = traceback.format_exc()

    def execute(self, code: str) -> str:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        p = multiprocessing.Process(target=self._execute_in_process, args=(code, return_dict))
        p.start()
        p.join(self.timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            return "Timeout Error: Execution exceeded 30 seconds."
        
        output = return_dict.get('output', '')
        error = return_dict.get('error', None)
        
        if error:
            return f"Execution Output:\n{output}\nError:\n{error}"
        elif not output.strip():
            return "Code executed successfully but printed no output. Please use print() to output the result."
        else:
            return f"Execution Output:\n{output}"

# ==========================================
# 2. TOÁN TỬ VÀ AGENT TƯ DUY
# ==========================================
SYSTEM_PROMPT = """You are an expert mathematical reasoning agent. 
To solve the math problem, you must write a Python script using the `sympy` or `math` libraries.
1. Think step-by-step.
2. Write Python code inside ```python ... ``` blocks to calculate intermediate or final results.
3. ALWAYS print the final result using `print()`. 
4. Analyze the output of your code.
5. The final answer must be a single non-negative integer. 
6. When you are certain of the final integer answer, output it in the format: \\boxed{answer}."""

def extract_code(text: str) -> str:
    pattern = r'```python(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else ""

def extract_boxed_answer(text: str) -> str:
    pattern = r'\\boxed\{([^{}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        num_matches = re.findall(r'\d+', matches[-1])
        if num_matches:
            return num_matches[-1]
    return ""

def solve_problem_agentic(problem: str, llm: LLM, sampling_params: SamplingParams, max_iterations=5):
    repl = PythonREPL(timeout=30)
    tokenizer = llm.get_tokenizer()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    
    for i in range(max_iterations):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        
        messages.append({"role": "assistant", "content": response_text})
        
        final_answer = extract_boxed_answer(response_text)
        if final_answer:
            return final_answer
            
        code = extract_code(response_text)
        if code:
            observation = repl.execute(code)
            messages.append({
                "role": "user", 
                "content": f"Observation from Python execution:\n```output\n{observation}\n```\nPlease continue."
            })
        else:
            messages.append({
                "role": "user", 
                "content": "You didn't write any Python code. Please write code to solve the problem, or provide the final answer in \\boxed{}."
            })
            
    # Fallback
    fallback_prompt = "You have run out of code execution iterations. Please provide your best guess for the final integer answer directly in \\boxed{} format without writing any more code."
    messages.append({"role": "user", "content": fallback_prompt})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    final_response = outputs[0].outputs[0].text
    
    final_answer = extract_boxed_answer(final_response)
    return final_answer if final_answer else "0"

# ==========================================
# 3. SELF-CONSISTENCY & MAJORITY VOTING
# ==========================================
def solve_with_majority_voting(problem: str, llm: LLM, num_samples=15) -> int:
    print(f"--- Solving Problem with {num_samples} Self-Consistency paths ---")
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=2048,
        stop=['```output']
    )
    
    answers = []
    for i in range(num_samples):
        ans_str = solve_problem_agentic(problem, llm, sampling_params)
        try:
            # Lấy số nguyên cuối cùng (Hệ thống Kaggle sẽ tự xử lý luật Modulo nếu có yêu cầu trong đề)
            ans_int = int(ans_str)
            answers.append(ans_int)
        except ValueError:
            pass
            
    if not answers:
        return 0
        
    vote_counts = Counter(answers)
    best_answer, count = vote_counts.most_common(1)[0]
    print(f"Selected Answer: {best_answer} (with {count} votes)")
    
    return best_answer

# ==========================================
# 4. KHỞI TẠO MODEL TOÀN CỤC (GLOBAL)
# ==========================================
print("Initializing LLM Engine...")
MODEL_PATH = "/kaggle/input/models/adriansaezmartinez/qwen2.5-math-7b-instruct-awq/transformers/default/1"
try:
    llm = LLM(
        model=MODEL_PATH,
        quantization="AWQ",
        tensor_parallel_size=2, # Nếu dùng T4x2 thì đổi thành 2
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    llm = None

# ==========================================
# 5. KAGGLE INFERENCE SERVER INTEGRATION
# ==========================================
def predict(test_df: pl.DataFrame, sample_prediction_df: pl.DataFrame) -> pl.DataFrame:
    """
    Hàm này được Kaggle Server gọi TỰ ĐỘNG cho MỖI bài toán.
    test_df: Bảng chứa 1 dòng đề bài (cột 'id', 'problem')
    sample_prediction_df: Bảng chứa 1 dòng mẫu (cột 'id', 'answer')
    """
    if llm is None:
        # Fallback an toàn nếu model sập
        return sample_prediction_df.with_columns(pl.lit(0).alias('answer'))

    # Lấy nội dung bài toán từ Polars DataFrame
    problem_text = test_df["problem"][0]
    problem_id = test_df["id"][0]
    
    print(f"\n{'='*50}\nProcessing Problem ID: {problem_id}\n{'='*50}")
    
    # Số lượng lấy mẫu có thể điều chỉnh dựa trên thời gian 9 tiếng / 50 bài
    NUM_SAMPLES = 15 
    final_answer = solve_with_majority_voting(problem_text, llm, num_samples=NUM_SAMPLES)
    
    # Ghi đè cột 'answer' bằng đáp án tìm được
    submission = sample_prediction_df.with_columns(pl.lit(final_answer).alias('answer'))
    return submission

if __name__ == "__main__":
    # Khởi chạy máy chủ suy luận của Kaggle
    inference_server = aimo_3_inference_server.AIMO3InferenceServer(predict)
    inference_server.serve()
