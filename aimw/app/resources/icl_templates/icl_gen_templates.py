template_dp_qa = """\
For each context, create up to {number_of_questions} different questions and answers that are specific to the context.
Questions and answers must cover all aspects of the context, and they must be consise and informative.
One question must be generic that covers the whole context.
Do not create questions that are not relevant to the context.

Format the output as JSON with the following keys:

queries_aspects: list of questions and answers about the context.


context: {context}
"""


template_qgen_agen = """\
Answer the questions using the given context. 
Format the output as JSON with the following keys:

queries_aspects: list of questions and answers about the context.

questions: {questions}
context: {context}
"""
