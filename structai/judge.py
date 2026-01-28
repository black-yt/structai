from .mp import multi_thread
from .utils import remove_tag, timeout_limit, parse_think_answer
from .io import print_once
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
    """
    A class for evaluating model answers against ground truth answers using multiple methods:
    Exact Match, Math Verify, and LLM-based Judge.
    """
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
        """
        Initialize the Judge.

        Args:
            api_key (str, optional): API Key. Defaults to `os.environ["LLM_API_KEY"]`.
            api_base (str, optional): Base URL. Defaults to `os.environ["LLM_BASE_URL"]`.
            model_version (str, optional): Model identifier for the LLM Judge. Default 'gpt-4.1'.
            system_prompt (str, optional): System prompt for the LLM Judge. Default 'You are a helpful assistant.'.
            max_tokens (int, optional): Maximum tokens for LLM generation. Default 10.
            temperature (float, optional): Sampling temperature for LLM. Default 0.
            http_client (httpx.Client, optional): Optional custom httpx client.
            headers (dict, optional): Optional custom headers.
            time_limit (int, optional): Timeout in seconds for LLM API calls. Default 60.
            max_try (int, optional): Number of retries for LLM API calls. Default 2.
            use_responses_api (bool, optional): Whether to use the Responses API format. Default False.
            prompt_tmp (str, optional): Template for the LLM Judge prompt. Defaults to `default_prompt_tmp`.
            use_tqdm (bool, optional): Whether to show a progress bar for batch processing. Default True.
            use_math_verify (bool, optional): Whether to use the `math_verify` library for evaluation. Default True.
            use_llm_judge (bool, optional): Whether to use an LLM for evaluation. Default True.
            llm_tags (dict, optional): Mapping of LLM output strings to scores. Default {"correct": 1, "incorrect": 0}.
            workers (int, optional): Number of threads for parallel processing. Default 100.
        """
        self.use_tqdm = use_tqdm
        self.use_math_verify = use_math_verify
        self.use_llm_judge = use_llm_judge
        self.workers = workers
        if use_llm_judge:
            self.judger = LLMAgent(api_key=api_key, api_base=api_base, model_version=model_version, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature, http_client=http_client, headers=headers, time_limit=time_limit, max_try=max_try, use_responses_api=use_responses_api)
            self.prompt_tmp = prompt_tmp
            self.llm_tags = {}
            for k, v in llm_tags.items():
                self.llm_tags[k.strip().lower()] = v


    def parse_short_answer(self, s: str):
        """
        Extracts the short answer from a potentially long model output.
        It handles chain-of-thought formats (e.g., <think>...</think>) if present.

        Args:
            s (str): The raw model output string.

        Returns:
            str: The extracted short answer.
        """
        if "</think>" in s or "<answer>" in s:
            try:
                _, short_answer = parse_think_answer(s)
            except:
                short_answer = s
        else:
            short_answer = s
        return short_answer


    def exact_match(self, short_model_answer: str, short_answer: str):
        """
        Performs a case-insensitive exact match comparison.

        Args:
            short_model_answer (str): The extracted model answer.
            short_answer (str): The ground truth answer.

        Returns:
            int: 1 if they match, 0 otherwise.
        """
        if short_model_answer.lower().strip() == short_answer.lower().strip():
            return 1
        else:
            return 0


    def math_verify(self, short_model_answer: str, short_answer: str):
        """
        Uses the `math_verify` library to check for mathematical equivalence.

        Args:
            short_model_answer (str): The extracted model answer.
            short_answer (str): The ground truth answer.

        Returns:
            int | None: 1 if equivalent, 0 if not, None if verification failed or library missing.
        """
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
            print_once(f"[===ERROR===][structai][judge.py][Judge.math_verify]{str(e)}\n")
            return None


    def llm_judge(self, question: str, model_answer: str, answer: str, solution: str=None):
        """
        Uses an LLM to judge the correctness of the model answer given the question and ground truth.

        Args:
            question (str): The question text.
            model_answer (str): The model's full answer.
            answer (str): The ground truth answer.
            solution (str, optional): The step-by-step solution (if available).

        Returns:
            int | None: The score (e.g., 1 or 0) based on `llm_tags`, or None if failed.
        """
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
        """
        Evaluates a single question dictionary using enabled methods (Exact Match, Math Verify, LLM Judge).
        Updates the dictionary with evaluation results.

        Args:
            ques_dict (dict): A dictionary containing 'question', 'answer', 'model_answer', and optionally 'solution'.

        Returns:
            dict: The updated dictionary with evaluation metrics.
        """
        question: str = ques_dict["question"]
        solution: str = ques_dict["solution"] if "solution" in ques_dict else None
        answer: str = ques_dict["answer"]
        model_answer: str = ques_dict['model_answer']

        short_answer = self.parse_short_answer(answer)
        
        exact_match_list = []
        math_verify_list = []
        math_verify_cache = {}
        llm_judge_list = []
        llm_judge_cache = {}

        model_answer_list = model_answer.split("<answer_split>")
        for model_answer in model_answer_list:
            short_model_answer = self.parse_short_answer(model_answer)
            exact_match_result = self.exact_match(short_model_answer, short_answer)
            exact_match_list.append(exact_match_result)

            if self.use_math_verify:
                if exact_match_result:
                    math_verify_result = 1
                    math_verify_cache[model_answer] = math_verify_result
                elif model_answer not in math_verify_cache:
                    math_verify_result = self.math_verify(short_model_answer, short_answer)
                    math_verify_cache[model_answer] = math_verify_result
                else:
                    # print('math_verify_cache Hit')
                    math_verify_result = math_verify_cache[model_answer]
                math_verify_list.append(math_verify_result)
            else:
                math_verify_result = None
                math_verify_list.append(math_verify_result)

            if self.use_llm_judge:
                if exact_match_result or math_verify_result:
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
        ques_dict["math_verify_list"] = math_verify_list
        ques_dict["llm_judge_list"] = llm_judge_list

        if exact_match_list[-1] == 1:
            ques_dict["exact_match"] = 1
        else:
            ques_dict["exact_match"] = 0
        if math_verify_list[-1] == 1:
            ques_dict["math_verify"] = 1
        else:
            ques_dict["math_verify"] = 0
        if llm_judge_list[-1] == 1:
            ques_dict["llm_judge"] = 1
        else:
            ques_dict["llm_judge"] = 0

        ##########################################################################
        if 1 in exact_match_list:
            ques_dict["exact_match_pass@k"] = 1
        else:
            ques_dict["exact_match_pass@k"] = 0
        if 1 in math_verify_list:
            ques_dict["math_verify_pass@k"] = 1
        else:
            ques_dict["math_verify_pass@k"] = 0
        if 1 in llm_judge_list:
            ques_dict["llm_judge_pass@k"] = 1
        else:
            ques_dict["llm_judge_pass@k"] = 0

        ##########################################################################
        if exact_match_list.count(1) == len(exact_match_list):
            ques_dict["exact_match_passall@k"] = 1
        else:
            ques_dict["exact_match_passall@k"] = 0
        if math_verify_list.count(1) == len(math_verify_list):
            ques_dict["math_verify_passall@k"] = 1
        else:
            ques_dict["math_verify_passall@k"] = 0
        if llm_judge_list.count(1) == len(llm_judge_list):
            ques_dict["llm_judge_passall@k"] = 1
        else:
            ques_dict["llm_judge_passall@k"] = 0

        if not self.use_math_verify:
            del ques_dict["math_verify_list"]
            del ques_dict["math_verify"]
            del ques_dict["math_verify_pass@k"]
            del ques_dict["math_verify_passall@k"]
        
        if not self.use_llm_judge:
            del ques_dict["llm_judge_list"]
            del ques_dict["llm_judge"]
            del ques_dict["llm_judge_pass@k"]
            del ques_dict["llm_judge_passall@k"]

        return ques_dict


    def __call__(self, ques_dict: Union[dict, list[dict]]):
        """
        Evaluates one or more question dictionaries using the configured evaluation methods (Exact Match, Math Verify, LLM Judge).

        This method processes the input dictionary (or list of dictionaries), extracts the model answer(s),
        and compares them against the ground truth answer using the enabled evaluation strategies.
        It supports multiple model answer samples separated by `<answer_split>`.

        Args:
            ques_dict (dict | list[dict]): A single dictionary or a list of dictionaries containing evaluation data.
                Each dictionary must contain the following keys:
                - "question" (str): The question text.
                - "answer" (str): The ground truth answer.
                - "model_answer" (str): The model's answer. If multiple samples are provided, they should be separated by `<answer_split>`.
                - "solution" (str, optional): The step-by-step ground truth solution.

        Returns:
            dict | list[dict]: The input dictionary (or list of dictionaries) updated with the following evaluation metrics:

            **Per-Sample Results (Lists):**
            - "exact_match_list" (list[int]): A list of 0s and 1s indicating whether each sample in `model_answer` exactly matches the ground truth (case-insensitive).
            - "math_verify_list" (list[int | None]): A list of 0s and 1s indicating mathematical equivalence for each sample (if `use_math_verify` is True).
            - "llm_judge_list" (list[int | None]): A list of 0s and 1s indicating correctness as judged by an LLM for each sample (if `use_llm_judge` is True).

            **Single-Sample Metrics (Based on the LAST sample):**
            - "exact_match" (int): 1 if the **last** sample is an exact match, 0 otherwise.
            - "math_verify" (int): 1 if the **last** sample is mathematically equivalent, 0 otherwise (if enabled).
            - "llm_judge" (int): 1 if the **last** sample is correct according to the LLM, 0 otherwise (if enabled).

            **Pass@k Metrics (At least ONE sample is correct):**
            - "exact_match_pass@k" (int): 1 if **any** sample in the list is an exact match, 0 otherwise.
            - "math_verify_pass@k" (int): 1 if **any** sample is mathematically equivalent, 0 otherwise (if enabled).
            - "llm_judge_pass@k" (int): 1 if **any** sample is correct according to the LLM, 0 otherwise (if enabled).

            **PassAll@k Metrics (ALL samples are correct):**
            - "exact_match_passall@k" (int): 1 if **all** samples are exact matches, 0 otherwise.
            - "math_verify_passall@k" (int): 1 if **all** samples are mathematically equivalent, 0 otherwise (if enabled).
            - "llm_judge_passall@k" (int): 1 if **all** samples are correct according to the LLM, 0 otherwise (if enabled).
        """
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
