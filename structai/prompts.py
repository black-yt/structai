llm_judge_closed_answer = """# Role
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


llm_judge_arena = """# Role
You are an impartial and rigorous evaluator. Your task is to compare two answers to the same open-ended question and determine which answer is better overall.

# Evaluation Criteria
1. **Relevance**: How well does the answer address the question being asked?
2. **Correctness & Soundness**: Are the claims, reasoning, or conclusions logically valid and factually sound within the context of the question?
3. **Completeness**: Does the answer cover the important aspects of the question without significant omissions?
4. **Clarity & Coherence**: Is the answer clearly structured, easy to understand, and logically organized?
5. **Insightfulness**: Does the answer demonstrate depth of understanding, useful reasoning, or meaningful insights appropriate for an open-ended question?

# Data
<question>
{question}
</question>

<answer_A>
{answer}
</answer_A>

<answer_B>
{model_answer}
</answer_B>

# Instruction
1. Carefully read the question and both answers.
2. Compare Answer A and Answer B using the evaluation criteria above.
3. Determine which answer is better overall. If one answer is clearly superior, select it.
4. If both answers are very similar in quality, choose the one that is slightly stronger based on the criteria, without defaulting to either side.

# Final Output Format
Output ONLY a single character: "A" or "B". Do not include any explanations, punctuation, or additional text."""

prompts = {
    "llm_judge_closed_answer": {
        "prompt_tmp": llm_judge_closed_answer,
        "llm_tags": {"correct": 1, "incorrect": 0}
    },
    "llm_judge_arena": {
        "prompt_tmp": llm_judge_arena,
        "llm_tags": {"A": 0, "B": 1}
    }
}