from .mp import multi_thread
from .utils import remove_tag, timeout_limit, parse_think_answer
from .llm_api import LLMAgent
from typing import Union


default_prompt_tmp = """# Role
You are a precise mathematical and logical evaluator. Your task is to compare a Model Answer against a Ground Truth Answer based on a given Question.

# Evaluation Criteria
1. **Numerical Equivalence**: Numbers must be mathematically equal even if formatted differently (e.g., 0.5 == 1/2, 1e-3 == 0.001, 2.0 == 2).
2. **Symbolic Equivalence**: Mathematical expressions should be equivalent (e.g., x + y == y + x).
3. **Unit Consistency**: If units are present, they must be compatible or correctly converted.
4. **Contextual Meaning**: If the answer is text-based, the core meaning must match the Ground Truth, ignoring minor paraphrasing or capitalization.

# Data
<question>
{question}
</question>

<answer>
{answer}
</answer>

<model_answer>
{model_answer}
</model_answer>

# Instruction
1. Mentally analyze if the Model Answer is mathematically or logically equivalent to the Ground Truth Answer.
2. Ignore formatting differences (like LaTeX vs. plain text) as long as the value is correct.
3. If the Model Answer is equivalent, the result is "correct".
4. If there is a numerical discrepancy or logical error, the result is "incorrect".

# Final Output Format
Output ONLY the string "correct" or "incorrect". Do not include any explanations, punctuation, or additional text."""


class Judge:
    def __init__(self, 
                api_key = None,
                api_base = None,
                model_version = 'gpt-4.1',
                system_prompt = 'You are a helpful assistant.',
                max_tokens = 10,
                temperature = 0,
                http_client = None,
                headers = None,
                time_limit = 60,
                max_try = 2,
                use_responses_api = False,
                prompt_tmp=default_prompt_tmp, 
                use_tqdm=True,
                use_math_verify=True,
                use_llm_judge=True,
                llm_tags={"correct": 1, "incorrect": 0},
                workers=100
        ):
        self.use_tqdm = use_tqdm
        self.use_math_verify = use_math_verify
        self.use_llm_judge = use_llm_judge
        self.workers = workers
        if use_llm_judge:
            self.judger = LLMAgent(api_key=api_key, api_base=api_base, model_version=model_version, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature, http_client=http_client, headers=headers, time_limit=time_limit, max_try=max_try, use_responses_api=use_responses_api)
            self.prompt_tmp = prompt_tmp
            self.llm_tags = llm_tags


    def parse_short_answer(self, s: str):
        if "</think>" in s or "<answer>" in s:
            try:
                _, short_answer = parse_think_answer(s)
            except:
                short_answer = s
        else:
            short_answer = s
        return short_answer


    def exact_match(self, short_model_answer: str, short_answer: str):
        if short_model_answer.lower().strip() == short_answer.lower().strip():
            return 1
        else:
            return 0


    def math_verify(self, short_model_answer: str, short_answer: str):
        @timeout_limit(5)
        def parse_and_verify(short_model_answer: str, short_answer: str):
            short_model_answer_parsed = parse(short_model_answer, parsing_timeout=None)
            short_answer_parsed = parse(short_answer, parsing_timeout=None)
            if verify(short_answer_parsed, short_model_answer_parsed, float_rounding=4, strict=False, allow_set_relation_comp=True, timeout_seconds=None):
                return 1
            return 0
        
        try:
            from math_verify import parse, verify
        except:
            print("Please install math_verify: pip install math-verify[antlr4_13_2]")
            return None
        
        try:
            return parse_and_verify(short_model_answer, short_answer)
        except Exception as e:
            print(f"[===ERROR===][structai][judge.py][Judge.math_verify]{str(e)}\n")
            return None


    def llm_judge(self, question: str, model_answer: str, answer: str, solution: str=None):
        if isinstance(solution, str) and solution and solution.lower() != "none":
            answer = solution + "\n\nFinal Answer: " + answer
        else:
            answer = answer
        answer = remove_tag(answer, r="\n\n")

        try:
            model_answer = remove_tag(model_answer)
            prompt = self.prompt_tmp.format(question=question, answer=answer, model_answer=model_answer)

            response = self.judger(prompt)
            assert isinstance(response, str), f"[LLM Judge response is not string][{response}]"
            response = response.strip().lower()
            assert response in list(self.llm_tags.keys()), f"[LLM Judge response not in {', '.join(list(self.llm_tags.keys()))}][{response}]"
            return self.llm_tags[response]
        except Exception as e:
            print(f"[===ERROR===][structai][judge.py][Judge.llm_judge]{str(e)}\n")
            return None


    def get_judge(self, ques_dict):
        question: str = ques_dict["question"]
        solution: str = ques_dict["solution"] if "solution" in ques_dict else None
        answer: str = ques_dict["answer"]
        model_answer: str = ques_dict['model_answer']

        short_answer = self.parse_short_answer(answer)
        
        exact_match_list = []
        math_veify_list = []
        math_veify_cache = {}
        llm_judge_list = []
        llm_judge_cache = {}

        model_answer_list = model_answer.split("<answer_split>")
        for model_answer in model_answer_list:
            short_model_answer = self.parse_short_answer(model_answer)
            exact_match_result = self.exact_match(short_model_answer, short_answer)
            exact_match_list.append(exact_match_result)

            if self.use_math_verify:
                if exact_match_result:
                    math_veify_result = 1
                    math_veify_cache[model_answer] = math_veify_result
                elif model_answer not in math_veify_cache:
                    math_veify_result = self.math_verify(short_model_answer, short_answer)
                    math_veify_cache[model_answer] = math_veify_result
                else:
                    # print('math_veify_cache Hit')
                    math_veify_result = math_veify_cache[model_answer]
                math_veify_list.append(math_veify_result)
            else:
                math_veify_result = None
                math_veify_list.append(math_veify_result)

            if self.use_llm_judge:
                if exact_match_result or math_veify_result:
                    llm_judge_result = 1
                    llm_judge_cache[model_answer] = llm_judge_result
                elif model_answer not in llm_judge_cache:
                    llm_judge_result = self.llm_judge(question, model_answer, answer, solution)
                    llm_judge_cache[model_answer] = llm_judge_result
                else:
                    # print('llm_judge_cache Hit')
                    llm_judge_result = llm_judge_cache[model_answer]
                llm_judge_list.append(llm_judge_result)
            else:
                llm_judge_result = None
                llm_judge_list.append(llm_judge_result)

        ques_dict["exact_match_list"] = exact_match_list
        ques_dict["math_veify_list"] = math_veify_list
        ques_dict["llm_judge_list"] = llm_judge_list

        if exact_match_list[-1] == 1:
            ques_dict["exact_match"] = 1
        else:
            ques_dict["exact_match"] = 0
        if math_veify_list[-1] == 1:
            ques_dict["math_veify"] = 1
        else:
            ques_dict["math_veify"] = 0
        if llm_judge_list[-1] == 1:
            ques_dict["llm_judge"] = 1
        else:
            ques_dict["llm_judge"] = 0

        ##########################################################################
        if 1 in exact_match_list:
            ques_dict["exact_match_pass@k"] = 1
        else:
            ques_dict["exact_match_pass@k"] = 0
        if 1 in math_veify_list:
            ques_dict["math_veify_pass@k"] = 1
        else:
            ques_dict["math_veify_pass@k"] = 0
        if 1 in llm_judge_list:
            ques_dict["llm_judge_pass@k"] = 1
        else:
            ques_dict["llm_judge_pass@k"] = 0

        ##########################################################################
        if exact_match_list.count(1) == len(exact_match_list):
            ques_dict["exact_match_passall@k"] = 1
        else:
            ques_dict["exact_match_passall@k"] = 0
        if math_veify_list.count(1) == len(math_veify_list):
            ques_dict["math_veify_passall@k"] = 1
        else:
            ques_dict["math_veify_passall@k"] = 0
        if llm_judge_list.count(1) == len(llm_judge_list):
            ques_dict["llm_judge_passall@k"] = 1
        else:
            ques_dict["llm_judge_passall@k"] = 0

        if not self.use_math_verify:
            del ques_dict["math_veify_list"]
            del ques_dict["math_veify"]
            del ques_dict["math_veify_pass@k"]
            del ques_dict["math_veify_passall@k"]
        
        if not self.use_llm_judge:
            del ques_dict["llm_judge_list"]
            del ques_dict["llm_judge"]
            del ques_dict["llm_judge_pass@k"]
            del ques_dict["llm_judge_passall@k"]

        return ques_dict


    def __call__(self, ques_dict: Union[dict, list[dict]]):
        if isinstance(ques_dict, dict):
            return self.get_judge(ques_dict)

        else:
            ques_dicts = [{"ques_dict": item} for item in ques_dict]
            ques_dicts = multi_thread(ques_dicts, self.get_judge, self.workers, self.use_tqdm)
            return ques_dicts


if __name__ == "__main__":
    import json
    ques_dicts = []
    ques_dicts.append({
        "question": "Bob's age?",
        "answer": "22",
        "model_answer": "21"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "answer": "22",
        "model_answer": "22"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "answer": "22",
        "model_answer": "22<answer_split>21"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "answer": "22",
        "model_answer": "20<answer_split>21"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "answer": "22",
        "model_answer": "22<answer_split>22"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "solution": "He was born in 2003, and today is 2025.",
        "answer": "22",
        "model_answer": "20<answer_split>Bob's age is 22"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "solution": "He was born in 2003, and today is 2025.",
        "answer": "22",
        "model_answer": "20<answer_split>20+2"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "solution": "He was born in 2003, and today is 2025.",
        "answer": "22",
        "model_answer": "20<answer_split>20+2<answer_split>20"
    })
    ques_dicts.append({
        "question": "Bob's age?",
        "solution": "He was born in 2003, and today is 2025.",
        "answer": "22",
        "model_answer": "20<answer_split>二十二<answer_split>20"
    })

    judge = Judge()
    judge_result = judge(ques_dicts[-1])
    print(json.dumps(judge_result, indent=4, ensure_ascii=False))

    judge_results = judge(ques_dicts)
    print(json.dumps(judge_results, indent=4, ensure_ascii=False))

    print("---------------------------------------------")

    judge = Judge(use_llm_judge=False)
    judge_result = judge(ques_dicts[0])
    print(json.dumps(judge_result, indent=4, ensure_ascii=False))

    judge_results = judge(ques_dicts)
    print(json.dumps(judge_results, indent=4, ensure_ascii=False))

    print("---------------------------------------------")

    judge = Judge(use_math_verify=False)
    judge_result = judge(ques_dicts[0])
    print(json.dumps(judge_result, indent=4, ensure_ascii=False))

    judge_results = judge(ques_dicts)
    print(json.dumps(judge_results, indent=4, ensure_ascii=False))