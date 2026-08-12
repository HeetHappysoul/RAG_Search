import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv()
## session state
if "Processed_file_name" not in st.session_state:
    st.session_state["Processed_file_name"] = None

## Part where the chunks are loaded in cache and it returns the embeddings of it


@st.cache_resource
def load_engine():
        ## This is the part of chunking of the static Sentences
    # text = "We are the flagship infrastructure engineering and construction company of the Shapoorji Pallonji group. We have a strong track record of executing numerous technologically complex EPC projects both within India and internationally.According to the Fitch Report, we are one of Indias largest international infrastructure construction companies, as per the 2023 ENR Top International Contractors rankings, based on International Revenue. In the last ten financial years and the six months ended September 30, 2023, we have successfully completed 76 projects across 15 countries with a total historic executed contract value of ₹522.20 billion."
    # split = text.split(".")
    # split.pop()
    # new_text = []

    # for i in range(0, len(split), 2):
    #     # Glue the 2 sentences together with a period and space, then add a final period
    #     chunk_string = ". ".join(split[i:i+2]) + "."
    #     print(chunk_string)
    #     new_text.append(chunk_string)


    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    # collection = chroma_client.create_collection(name = "My_first_rag")

    # embeddings = model.encode(new_text)
    # id = [f"id{i}" for i in range(len(new_text))]
    # collection.add(embeddings = embeddings.tolist(), documents = new_text, ids = id)


    return model,chroma_client

##Handling if the pdf is already uplaoded or not


## Home page things 



model,chroma_client = load_engine()
st.success("Engine loaded successfully ")
st.title("This is an DHRP Expert")

## Handling the pdf and extraction of text
file = st.file_uploader("Enter the PDF for expert to read",type = "pdf")
if file:

    if st.session_state["Processed_file_name"]!= file.name:
        st.info(f"Processing new file : {file.name}")
        ## wipe the data
        try :
            chroma_client.delete_collection(name = "My_first_rag")
        except Exception:
            pass

        collection = chroma_client.create_collection(name = "My_first_rag")

        text = ""
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()

        st.write(text[:500])


    ## Chunking logically using semantic aggregation

        master_text = text.split(".")
        master_text.pop()
        current_chunk = ""
        final_chunk = []
        current_chunk +=master_text[0]
        for i in range(1,len(master_text)):
            x = model.encode(master_text[i-1])
            y = model.encode(master_text[i])
            similarity = np.dot(x,y)
            if similarity >= 0.85:
                current_chunk +=". " + master_text[i]
            else:
                final_chunk.append(current_chunk)
                current_chunk = master_text[i]
        final_chunk.append(current_chunk)

        embeddings = model.encode(final_chunk)






    ## adding the final chunk to the db

        id = [f"id{i}" for i in range(len(final_chunk))]


        collection.add(embeddings = embeddings.tolist(),documents = final_chunk, ids = id)

        st.session_state["Processed_file_name" ] = file.name
        st.success("file processed sucessfully")






## QUery 

user_query = st.text_input("Ask the question to the DHRP Expert")
is_clicked = st.button("Ask to Expert")






## This is the logic of extracting data and explaining API Thing.




if is_clicked:

    n_result = 4
    final_ans = ""

    collection = chroma_client.get_collection(name = "My_first_rag")
    q_embed = model.encode(user_query)
    result = collection.query(query_embeddings =q_embed.tolist() ,n_results=n_result)
    ans = "\n\n".join(result['documents'][0])

    
    # The client automatically securely pulls the key from your .env file
    my_key = os.getenv("GEMINI_API_KEY")
    # print("DEBUG - MY KEY IS:", my_key)
    client = genai.Client(api_key = my_key)
    prompt = f"""You are a precise assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, reply with "I do not know based on the provided document.Also if you don't find anything about the use_query then you say you don't have enough data to anser the question relevantly"

    Context:
    {ans}

    Question:
    {user_query}"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
)
    st.write(interaction.output_text)
    st.info(f"🔍 DEBUG - RETRIEVED CONTEXT: {ans}")
