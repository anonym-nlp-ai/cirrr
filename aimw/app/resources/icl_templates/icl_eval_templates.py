
comprehensiveness_template = """
    Analyse the comprehensiveness of the generated question-answer pairs in relation to the given context. Follow this structured approach:\n
        1. Assess if the question-answer pairs address the main ideas and important details in the context.\n
        2. Evaluate whether the questions cover relevant aspects of the context and prompt deeper understanding.\n
        3. Determine the correctness and completeness of the answers based on the information in the context.\n
        4. Assess the overall flow and logical connection between the questions and answers.\n\n
    
    Provide a score for each criterion (1-5, with 5 being the highest), along with a thorough explanation of your reasoning. Conclude with an overall assessment of the question-answer pair's comprehensiveness.\n\n
    
    Context: {context} \n
    Generated question-answer Pairs: {question_answer_list}
"""


faithfulness_template = """
   Analyse the faithfulness and potential for hallucination in the generated QA pairs in relation to the given context. Follow this structured approach:\n
        1. Assess whether the questions and answers accurately reflect the information provided in the context.\n
        2. Determine if any claims in the answers overstate or embellish the information from the context.\n
        3. Determine if the answers contradict or deviate from the facts presented in the context.\n
        4. Evaluate whether the questions and answers are well-supported by evidence from the context.\n
        5. Assess whether the questions and answers, even if not explicitly stated, are reasonable inferences based on the context.\n
        6. Assess if any facts in the context are distorted or presented misleadingly in the answers.\n\n
    
    Provide a score for each criterion (1-5, with 5 being the highest), along with a thorough explanation of your reasoning, including specific examples from the context and generated QA pairs. Conclude with an overall assessment of the QA pair's faithfulness and potential for hallucination.\n\n
    
    Context: {context} \n
    Generated question-answer Pairs: {question_answer_list}
"""