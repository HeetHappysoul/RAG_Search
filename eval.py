from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name = "Eval")


text = "" 
file = "Afcons_Concall_Q1FY2026.pdf"
reader = PdfReader(file)
for page in reader.pages:
    text += page.extract_text()
master_text = text.split(".")
master_text.pop()
current_chunk = "" 
final_chunk = []
current_chunk += master_text[0]
for i in range(1,len(master_text)):
    x = model.encode(master_text[i-1])
    y = model.encode(master_text[i])
    similarity = np.dot(x,y)
    if similarity >=0.85:
        current_chunk += ". "+master_text[i]
    else:
        final_chunk.append(current_chunk)
        current_chunk = master_text[i]
final_chunk.append(current_chunk)


embeddings = model.encode(final_chunk)
# print(embeddings)

id = [f"id{i}"  for i in range(len(final_chunk))]
collection = chroma_client.create_collection(name = "eval")
collection.add(embeddings = embeddings, documents = final_chunk, ids = id)

## questionaire
questionnaire = [


    {
        "id": "id1",
        "question": "What was Afcons' total income in Q1 FY26?",
        "expected_answer": "INR 3,419 crores.",
        "ground_truth_evidence": "Afcons reported a total income of INR3,419 crores in Q1 FY26."
    },

    {
        "id": "id2",
        "question": "What was Afcons' EBITDA margin in Q1 FY26?",
        "expected_answer": "13%.",
        "ground_truth_evidence": "The EBITDA margin for the quarter improved by 140 basis points, reaching 13%."
    },

    {
        "id": "id3",
        "question": "What EBITDA margin guidance did Afcons maintain for the year?",
        "expected_answer": "Around 11%.",
        "ground_truth_evidence": "Our guidance for EBITDA will remain around 11%."
    },

    {
        "id": "id4",
        "question": "What was Afcons' unexecuted order book as of June 30?",
        "expected_answer": "INR 35,311 crores.",
        "ground_truth_evidence": "As on 30th June, our unexecuted order book stands at INR35,311 crores."
    },

    {
        "id": "id5",
        "question": "What annual turnover growth is Afcons expecting for FY26?",
        "expected_answer": "20% to 25%.",
        "ground_truth_evidence": "We are confident of achieving our annual turnover growth of 20% to 25% in this financial year."
    },

    # {
    #     "id": "Q06",
    #     "question": "What is Afcons' addressable project pipeline for the next two years?",
    #     "expected_answer": "Approximately INR 3.35 lakh crores.",
    #     "ground_truth_evidence": "Our addressable project pipeline for the next two years is valued at approximately INR3.35 lakh crores."
    # },

    # {
    #     "id": "Q07",
    #     "question": "What amount is pending in UP for the Jal Jeevan Mission?",
    #     "expected_answer": "Approximately INR 422 crores at the gross level.",
    #     "ground_truth_evidence": "It is roughly around INR422 crores at the gross level."
    # },

    # {
    #     "id": "Q08",
    #     "question": "How much advance does Afcons hold against the UP Jal Jeevan Mission?",
    #     "expected_answer": "INR 87 crores.",
    #     "ground_truth_evidence": "INR87 crores we are holding advance."
    # },

    # {
    #     "id": "Q09",
    #     "question": "How long does Afcons normally expect to receive the order after being declared L1 for the Croatia project?",
    #     "expected_answer": "Within 120 days.",
    #     "ground_truth_evidence": "Normally, from the time they declare L1, within 120 days, they are supposed to give the order."
    # },

    # {
    #     "id": "Q10",
    #     "question": "What is Afcons' planned full-year capex?",
    #     "expected_answer": "Around INR 1,100 crores.",
    #     "ground_truth_evidence": "For the full year capex we have planned around INR1,100 crores."
    # },

    # {
    #     "id": "Q11",
    #     "question": "How much capex did Afcons incur in Q1?",
    #     "expected_answer": "Close to INR 50 crores.",
    #     "ground_truth_evidence": "In Q1, we have done close to INR50 crores of capex."
    # },

    # {
    #     "id": "Q12",
    #     "question": "What is Afcons' current net debt and net debt-to-equity ratio?",
    #     "expected_answer": "Net debt is below INR 2,500 crores and net debt-to-equity is around 0.46.",
    #     "ground_truth_evidence": "The net debt is sub INR2,500 crores and on net debt to equity basis, it is around 0.46."
    # },

    # {
    #     "id": "Q13",
    #     "question": "What is Afcons' order flow guidance for FY26?",
    #     "expected_answer": "INR 20,000 crores.",
    #     "ground_truth_evidence": "Our order flow guidance for the full year is INR20,000 crores."
    # },

    # {
    #     "id": "Q14",
    #     "question": "What are the current proportions of interest-free and interest-bearing advances?",
    #     "expected_answer": "63% is interest-free and 37% is interest-bearing.",
    #     "ground_truth_evidence": "Today it is 37% is interest bearing and 63% is interest free."
    # },

    # {
    #     "id": "Q15",
    #     "question": "What is Afcons' average borrowing cost?",
    #     "expected_answer": "Around 9%.",
    #     "ground_truth_evidence": "On our bank borrowing, the average borrowing is around 9%."
    # },

    # {
    #     "id": "Q16",
    #     "question": "What is the approximate average interest cost on interest-bearing advances?",
    #     "expected_answer": "Around 7% to 10%.",
    #     "ground_truth_evidence": "Some of the interest bearing advances, it varies from 7% to 10%."
    # },

    # {
    #     "id": "Q17",
    #     "question": "How is Afcons' project pipeline divided across its major segments?",
    #     "expected_answer": "Around INR 1.4 lakh crores urban infrastructure, INR 80,000 crores hydro/underground/water, INR 70,000 crores surface transport, and INR 46,000 crores marine.",
    #     "ground_truth_evidence": "The largest chunk continues to be the urban infrastructure project, around INR1.4 lakh crores... hydro, underground, and water ... around INR80,000 crores... surface transport ... around INR70,000 crores... remaining marine is around INR46,000 crores."
    # },

    # {
    #     "id": "Q18",
    #     "question": "What is the domestic versus overseas split of Afcons' pending pipeline?",
    #     "expected_answer": "Approximately two-thirds domestic and one-third overseas.",
    #     "ground_truth_evidence": "Overseas and domestic, it's around one-third and two-third is the breakup in the pending pipeline."
    # },

    # {
    #     "id": "Q19",
    #     "question": "Why is the high-speed rail project facing uncertainty?",
    #     "expected_answer": "Because TBM delivery has been delayed; other work such as NATM tunneling and shafts is progressing.",
    #     "ground_truth_evidence": "On TBM, as of now, there is uncertainty prevailing... NATM portion of the tunneling... is already completed."
    # },

    # {
    #     "id": "Q20",
    #     "question": "What is Afcons' approach to entering newer business segments?",
    #     "expected_answer": "Afcons prefers entering related or adjacent infrastructure segments rather than completely unrelated businesses.",
    #     "ground_truth_evidence": "These growth always in the related sectors... newer areas ... but not in unrelated areas."
    # }
]


## Extracting question from questionaire and passing it to llm to find the answer 
## Two LLM's will be used one to generate and one to check the answer
score = 0
evaluated_questions = 5
for question in questionnaire:

    query = question["question"]

    q_embed = model.encode(query)
    result = collection.query(query_embeddings = q_embed.tolist(), n_results = 3)
    # id_query = result["ids"]
    ans = "\n\n" .join(result['documents'][0])
    my_key = os.getenv("GEMINI_API_KEY")
        # print("DEBUG - MY KEY IS:", my_key)
    client = genai.Client(api_key = my_key)
    prompt = f"""You are a precise assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, reply with "I do not know based on the provided document.Also if you don't find anything about the use_query then you say you don't have enough data to anser the question relevantly"

    Context:
    {ans}

    Question:
    {query}"""

    interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
    )

    llm_op = interaction.output_text


    ## passing responses to the second llm to get either right or wrong.
    my_key2  = os.getenv("GEMINI_API_KEY2")
    client = genai.Client(api_key = my_key2)
    prompt = f""" Your simple job is to evaluate to act as an evaluator. Now this is a RAG project that I am working on. So what you have to do is you will get the response that was generated by the LLM previous to you. So your job is to evaluate the answer that I will send you as the. LL miss answer. I'm gonna be sending you the LLM answer along with the question, along with the expected answer that the LLM should generate. And maybe some different things. Now what your final job here is that, your final job is that. You have to check whether the expected answer and the LLM answer matches or not. They. The words can be here or there, but it should make the same meaning you. It is an DHRP project wherein there will be numbers as well. So you have to make sure that numbers match. Also. Also if you look at this thing your. Primary job is just to evaluate and you have to just respond in either zero or one. You have also been sent the "ans_retrieved_by_algo" i.e being retrived by the algo as well as the "ground_truth" i.e this is the thing that is correct based on the evaluation questionaire prepared !!!Strict instruction = That's it. If the answer matches and makes sense of the expected answer, the LLM's answer and the expected answer makes sense, you'll reply 1, and if there is something wrong, you'll reply 0 i.e only one numberical value either 0 or one.
     
    llm_answer = {llm_op}

    question = {query}

    expected_answer = {question["expected_answer"]}

    ans_retrieved_by_algo = {ans}

    ground_truth = {question["ground_truth_evidence"]}
        
     """

    interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
    )
    llm_op_2 = interaction.output_text
    print(llm_op_2)

    score += int(llm_op_2)

percentage = (score/evaluated_questions)*100

print(percentage)







