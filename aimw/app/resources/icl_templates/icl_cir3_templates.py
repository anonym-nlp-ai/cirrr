from langchain.prompts import PromptTemplate


classifier_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the document classifier Agent, you are an expert at analysing and understanding documents. \
    
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Conduct a comprehensive analysis of the input document provided, then classify its main topic and identify up to {M} subtopics it covers.\
    The output must be concise and formatted as JSON with the following key:\n

    subtopics: [list of subtopics starting by the main topic]\n\n
    
    Example demonstrations:\
        Example 01:\n
        document: At first glance, life insurance and annuities might seem like opposites. Life insurance is primarily used to pay your heirs when you pass away. An annuity grows your savings and pays you income while you're still alive. However, some life insurance policies let you build savings while alive, and annuities can include a death benefit payment.\n
        subtopics: [finance, life insurance, annuity, saving]\n

        Example 02:\n
        document: Homeowners insurance and mortgage insurance are both types of insurance that can add to the cost of owning property, and you're likely to encounter both during the mortgage process. However, that's where their similarity ends. Homeowners insurance protects your home and its contents, while mortgage insurance, also called private mortgage insurance (PMI), protects your mortgage lender in case you default on your mortgage payments.\n
        {{"subtopics": [finance, homeowners insurance, mortgage insurance, mortgage]\}}


        Example 03:\n
        document: Non-qualified tax deferred annuities accumulate tax deferred and upon distributions are taxed as ordinary income at the effective tax bracket rate of the annuity policy owner. Basis is distributed tax free . Qualified annuity distributions are taxed as ordinary income at the effective tax bracket rate of the annuity policy owner . There is no basis.\n
        {{"subtopics": [finance, annuity, tax]\}} \n\n

        Do not generate anything except the JSON.

    document:\n {document} \n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["M", "document"],
)


moderator_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the moderator agent. Your role is to supervise a team of agents.\
    You have been provided with a list of subtopics and a document. Your role is to instruct a group of {M} writer agents to generate a comprehensive and faithful set of question-answer pairs given the input document. \
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Genererate a set of instructions to be given to the writer agents based on the following json format:

    {{writers_instructions: 1. assign perspective to the writer i based on the subtopic i, then\
        2. Instruct the writer i to analyse the document in-depth from their assigned perspective. \
        3. Instruct the writer i to generate up to {N} question-answer pairs that are directly grounded in the document's content.\
        4. Instruct the writer i to reflect on their question-answer pairs with respect to the input document. If writer i agrees that no refinements are needed, they should return "agreement" as feedback, otherwise "continue".\
        5. Instruct the writers to format the output as JSON with the following keys:
            question_answer_list: [list of refined question-answer pairs] \n
            feedback: [list of new feedback] \n
            status: continue or agreement \n
            perspective: perspective\n
    }}


    subtopics: {subtopics} \n
    document: {document} \n
    Number of question-answer pairs to be generated: {N} \n\n

    Make sure to return no comments, but ONLY a valid JSON.

    Demonstration examples:{moderator_examplar}

    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["M", "document", "subtopics", "N", "moderator_examplar"],
)


moderator_examplar = [
    {
        "writer_id": 1,
        "perspective": "tax",
        "task": "Generate 5 question-answer pairs that are directly grounded in the document's content from the perspective of a tax expert.",
        "instructions": [
            "1. Analyze the document in-depth from your assigned perspective.",
            "2. Generate up to 5 question-answer pairs that are directly grounded in the document's content.",
            "3. Reflect on your question-answer pairs with respect to the input document. If you agree that no refinements are needed, return 'agreement' as writer_status, otherwise return 'continue', and output your feedback as 'writer_feedback'.",
            "4. Format your output as JSON with the following keys: 'question_answer_list', 'writer_feedback', 'writer_status', and 'perspective'.",
        ],
    },
    {
        "writer_id": 2,
        "perspective": "finance",
        "task": "Generate 5 question-answer pairs that are directly grounded in the document's content from the perspective of a finance expert.",
        "instructions": [
            "1. Analyze the document in-depth from your assigned perspective.",
            "2. Generate up to 5 question-answer pairs that are directly grounded in the document's content.",
            "3. Reflect on your question-answer pairs with respect to the input document. If you agree that no refinements are needed, return 'agreement' as writer_status, otherwise return 'continue', and output your feedback as 'writer_feedback'.",
            "4. Format your output as JSON with the following keys: 'question_answer_list', 'writer_feedback', 'writer_status', and 'perspective'.",
        ],
    },
    {
        "writer_id": 3,
        "perspective": "annuity",
        "task": "Generate 5 question-answer pairs that are directly grounded in the document's content from the perspective of an annuity expert.",
        "instructions": [
            "1. Analyze the document in-depth from your assigned perspective.",
            "2. Generate up to 5 question-answer pairs that are directly grounded in the document's content.",
            "3. Reflect on your question-answer pairs with respect to the input document. If you agree that no refinements are needed, return 'agreement' as writer_status, otherwise return 'continue', and output your feedback as 'writer_feedback'.",
            "4. You must format your output as JSON with the following keys: 'question_answer_list', 'writer_feedback', 'writer_status', and 'perspective'.",
        ],
    },
]


writer_initial_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert writer agent.\
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {moderator_prompt} \n\n


    document:{document}

    Do not generate anything except the JSON.
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["document", "moderator_prompt"],
)

writers_refinement_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert writer agent.\
    You have been provided with a document, feedback list, and a list of generated set of question-answer pairs. Both lists of feedback and question-answer pairs are the output from a collective effort of a group of writers (including yourself).\
    You role is to refine the list of the generated question-answer pairs based on:\n
        1. your perspective,\
        2. the feedback provided by the group of writers and the curmudgeon agent.\n
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Review and refine all generated question-answer pairs (as 'new_question_answer_list') based on the provided feedback from other writer agents. If you don't agree with the provided feedback from the writer, you must reflect on the input and write a new list of feedback (as 'new_writer_feedback'), while ensuring that all question-answer pairs: \n
        Include and accurately represent your assigned subtopic.\
        Are well-supported by the document's content.\
        Offer diverse and comprehensive coverage of the subtopic within the context of the document.\
        Are semantically distinct, and are not redundant.\n
    
    If you agree that no refinements are needed, return 'agreement' as 'writer_status', otherwise 'continue', and output your new feedback as 'new_writer_feedback'.\
    Output the refined list of question-answer pairs along with your feedback.\
    Format the output as JSON with the following keys:\
        new_question_answer_list: [list of refined question-answer pairs]\
        new_writer_feedback: [list of new feedback]\
        writer_status: continue or agreement.\n\n
        perspective: your perspective


    The following are the input:\n
        document: {document} \n
        old_question_answer_list: {old_question_answer_list} \n
        old_writer_feedback: {old_writer_feedback} \n
        curmudgeon_feedback: {curmudgeon_feedback} \n
        perspective: {perspective}

    Do not generate anything except the JSON.
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=[
        "document",
        "old_question_answer_list",
        "old_writer_feedback",
        "curmudgeon_feedback",
        "perspective"
    ],
)

curmudgeon_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the curmudgeon agent. Your role is to critic and provide constructive feedback to a group of agents.\

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Your task is to review and assess the quality of the generated question-answer pairs in relation to the original document, and taking in consederation your previous feedback (old_curmudgeon_feedback). Then, you select to {N} question-answer pairs based on the following instructions, and do not forget to add your feedback to each selected question-answer pair to ensure that they:\n
        1. Are both comprehensive, diverse, grounded in the document, and free from assumptions.\n
        2. Are both semantically distinct, rather than paraphrases or redundant instances.\n
        3. Are both diverse. To assess diversity and assess redundancy, use your judgement to determine whether to provide feedback based on your own knowledge, scores from an external evaluation tool, or both. The evaluation tool scores quantify the diversity of (a) generated questions (score\_1), (b) generated answers (score\_2), and (c) concatenated answers and input document (score\_3). The objective of your feedback on diversity is to minimise score\_3, while maximising score\_1 and score\_3. To call the tool return tool_calling as true. You can call the tool only once.\n\n

    However, if you are satisfied with the quality of the input set of question-answer pairs, return 'agreement' and the set of selected question-answer pairs without feedback, otherwise, instruct the writers to begin another round of generation and refinement by returning your selected question-answer pairs, your feedback and the status 'curmudgeon_status' as 'continue'.\n


    Format the output as JSON with the following keys:\n

        new_curmudgeon_feedback: [append your feedback to each selected question-answer pair using the key: curmudgeon_feedback]
        curmudgeon_status: continue or agreement. \n\n

    If the input variable 'last_iteration' is set to 'True', then, return the selected question-answers only, and remove 'curmudgeon_feedback'.\n


    The following are the input:\n
        document: {document} \n
        question_answer_list: {question_answer_list} \n
        old_curmudgeon_feedback: {old_curmudgeon_feedback} \n
        diversity_scores: {diversity_scores} \n
        last_iteration: {last_iteration} \n\n


    Make sure to return only a valid JSON, and no extra comments.

    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=[
        "document",
        "question_answer_list",
        "old_curmudgeon_feedback",
        "diversity_scores",
        "last_iteration"
    ],
)
