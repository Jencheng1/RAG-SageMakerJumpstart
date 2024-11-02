import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader
import json
import boto3
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from sagemaker.jumpstart.model import JumpStartModel
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

sagemaker_client = boto3.client('sagemaker')
runtime_client = boto3.client('sagemaker-runtime')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
@st.cache_resource
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        print(response_json)
        return response_json["generated_text"]
        #return response_json

def get_conversation_chain(vectorstore,endpoint_name):
    client = boto3.client(
                "sagemaker-runtime",
                region_name="us-west-2",
    )
    content_handler = ContentHandler()
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        client=client,
        model_kwargs={
                        "temperature": 0.7,
                        "max_new_tokens": 1000,
                        "top_p": 0.9,
                        "stop": "<|eot_id|>"
        },
        content_handler=content_handler,
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    memory.clear()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    output_str = ''
    for d in response['source_documents']:
        output_dict = d.metadata
        if output_dict.get('filename') is not None:
            output_str = '\n'+output_dict.get('filename')+output_str

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content+output_str), unsafe_allow_html=True)

def deploy_model_and_wait(model_id, instance_type):
    #role = get_execution_role()
    jumpstart_model = JumpStartModel(model_id=model_id)
    st.write(f"Deploying model: {model_id} with {instance_type}")
    #predictor = jumpstart_model.deploy(initial_instance_count=1, instance_type=instance_type)
    predictor = jumpstart_model.deploy(accept_eula=False, instance_type =instance_type )

    endpoint_name = predictor.endpoint_name

    # Check endpoint status
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        if status == 'InService':
            st.success(f"Endpoint {endpoint_name} is InService!")
            return endpoint_name
        elif status == 'Failed':
            st.error(f"Endpoint creation failed.")
            return None
        else:
            st.write(f"Endpoint status: {status}. Checking again in 30 seconds...")
        time.sleep(30)

def deploy_model_with_ttl(model_id, instance_type, hours_to_keep):
    #role = get_execution_role()

    # Calculate the TTL timestamp based on the input hours
    delete_after_time = (datetime.utcnow() + timedelta(hours=hours_to_keep)).isoformat() + 'Z'

    # Create the JumpStart model object
    jumpstart_model = JumpStartModel(model_id=model_id)

    # Deploy the model with the specified instance type
    st.write(f"Deploying model: {model_id} with {instance_type}")
    predictor = jumpstart_model.deploy(initial_instance_count=1, instance_type=instance_type)

    endpoint_name = predictor.endpoint_name

    # Add a TTL tag to the endpoint with the delete time
    tags = [
        {
            'Key': 'DeleteAfter',
            'Value': delete_after_time
        }
    ]

    # Attach the TTL tag to the endpoint
    sagemaker_client.add_tags(
        ResourceArn=f'arn:aws:sagemaker:{boto3.Session().region_name}:{boto3.client("sts").get_caller_identity()["Account"]}:endpoint/{endpoint_name}',
        Tags=tags
    )

    # st.success(f"Endpoint {endpoint_name} created and tagged for deletion after {hours_to_keep} hours.")
    # return endpoint_name
        # Check endpoint status
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        if status == 'InService':
            st.success(f"Endpoint {endpoint_name} is InService!")
            return endpoint_name
        elif status == 'Failed':
            st.error(f"Endpoint creation failed.")
            return None
        else:
            st.write(f"Endpoint status: {status}. Checking again in 30 seconds...")
        time.sleep(30)

def get_sagemaker_endpoints():
    response = sagemaker_client.list_endpoints(
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    endpoints = [ep['EndpointName'] for ep in response['Endpoints'] if ep['EndpointStatus'] == 'InService']
    return endpoints


def delete_sagemaker_endpoint(endpoint_name):
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        st.success(f"Deleted endpoint: {endpoint_name}")
    except Exception as e:
        st.error(f"Error deleting endpoint: {e}")

        
def query_endpoint(payload,endpoint_name):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType="application/json", 
        Body = json.dumps(payload).encode("utf-8")
                )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response


def format_instructions(instructions: List[Dict[str, str]]) -> List[str]:
    """Format instructions where conversation roles must alternate user/assistant/user/assistant/..."""
    prompt: List[str] = []
    for user, answer in zip(instructions[::2], instructions[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (instructions[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)


    
def sentiment_analysis(response:str,aspect:str, prompt:str,endpoint_name:str,client):

    #client = boto3.client("runtime.sagemaker")
    prompt_final = prompt.format(response,aspect)    
    payload =  {
        "inputs": prompt_final,
        "parameters": {
            "max_new_tokens": 2,
        },
    }
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=json.dumps(payload).encode("utf-8")
    )
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions["generated_text"]

    return generated_text

def categrize_analysis(response:str,prompt:str,endpoint_name:str):

    client = boto3.client("runtime.sagemaker")
    prompt_final = prompt.format(response)    
    payload =  {
        "inputs": prompt_final,
        "parameters": {
            "max_new_tokens": 5,
        },
    }
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=json.dumps(payload).encode("utf-8")
    )
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions["generated_text"]

    return generated_text

def split_func_1(Aspects:str,delimiter:str):
    Aspect_ls=[]
    Aspect_ls=Aspects.split(delimiter)
    if len(Aspect_ls)>1:
        Aspect=Aspect_ls[1]
    else:
        Aspect=''
    return Aspect

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Docs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "submitted" not in st.session_state:
        st.session_state.submitted=False
    if "categrize_submitted" not in st.session_state:
        st.session_state.categrize_submitted=False   
    if "sentiment_submitted" not in st.session_state:
        st.session_state.sentiment_submitted=False
        
    st.title("Unstructured Data Analysis with AWS Sagemaker Endpoint")
    #with st.sidebar:
    st.sidebar.title("Use Existing Endpoint")
    with st.sidebar.form(key="exist_form"):
        #endpoint_name=st.text_input("Enter Endpoint:")
        endpoints = get_sagemaker_endpoints()
        endpoint_name = st.selectbox('Select Endpoint:', endpoints)

        # Display the selected endpoint
        st.write(f"Selected SageMaker Endpoint: {endpoint_name}")
        existing_submit = st.form_submit_button(label="Checking Exisitng SageMaker Endpoint")
    st.sidebar.title("Create SageMaker New Endpoint")
    with st.sidebar.form(key="deploy_form"):
        #model_id = st.text_input("Enter SageMaker JumpStart Model ID", value="huggingface-llm-mistralai-mixtral-8x22B-instruct-v0-1")
        model_id = st.selectbox("Enter SageMaker JumpStart Model ID",options=["huggingface-llm-mixtral-8x7b-instruct","huggingface-llm-mistralai-mixtral-8x22B-instruct-v0-1",'huggingface-llm-mistral-7b-instruct-v3'])
        hours_to_keep = st.number_input("How many hours to keep the endpoint?", min_value=1, max_value=72, value=2)
        instance_type = st.selectbox("Select Instance Type", options=["ml.g5.48xlarge","ml.p4d.24xlarge","ml.g5.24xlarge","ml.g5.xlarge", "ml.p5.48xlarge"])
        deploy_submit = st.form_submit_button(label="Create SageMaker Endpoint")
    # Process SageMaker deployment if form is submitted
    st.sidebar.title("Cleanup Endpoint")
    with st.sidebar.form(key="clean_form"):
        #endpoint_name=st.text_input("Enter Endpoint:")
        endpoints = get_sagemaker_endpoints()
        endpoint_name = st.selectbox('Select Endpoint:', endpoints)

        # Display the selected endpoint
        st.write(f"Selected SageMaker Endpoint: {endpoint_name}")
        clean_submit = st.form_submit_button(label="Deleting SageMaker Endpoint")
    if deploy_submit:
        st.write(f"Create sagemaker endpoint")
        with st.spinner("Creating sagemaker endpoint..."):
            if model_id and instance_type and hours_to_keep:
            #if model_id and instance_type:
                #endpoint_name = deploy_model_and_wait(model_id, instance_type)
                endpoint_name = deploy_model_with_ttl(model_id, instance_type, hours_to_keep)
                if endpoint_name:
                    st.session_state['endpoint_name'] = endpoint_name
                    st.success(f"Endpoint '{st.session_state['endpoint_name']}' is active and will be deleted after {hours_to_keep} hours.")
            else:
                st.error("Please enter all required details.")

    if existing_submit:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            if status == 'InService':
                st.session_state['endpoint_name'] = endpoint_name
                st.success(f"Endpoint {endpoint_name} is InService!")
            else:
                st.write(f"Endpoint {endpoint_name} is not ready for use. Please create a new endpoint.")

    if clean_submit:
        response= delete_sagemaker_endpoint(endpoint_name)
        st.success(f"Endpoint {endpoint_name} has been deleted")

    st.sidebar.title("Upload File And Perform Data Analysis")
    with st.sidebar.form(key="vector_store_form"):
        st.subheader("Your documents")
        uploaded_file = st.file_uploader(
            "Upload your Docs:",
            accept_multiple_files=True
        )
        vector_store_submit = st.form_submit_button(label="RAG Chatbot")
        summarize_submit = st.form_submit_button(label="Summarize") 
        categrize_submit = st.form_submit_button(label = "Categrize")
        sentiment_submit = st.form_submit_button(label = "Sentiment Analysis")
        
        
    if summarize_submit and uploaded_file:
        st.session_state.submitted = False
        with st.form(key="summary_form"):
            def submitted():
                st.session_state.submitted = True
                st.session_state.col 
                st.session_state.prompt
            if ".pdf" in uploaded_file[0].name:
                raw_text= get_pdf_text(uploaded_file)
            elif ".csv" in uploaded_file[0].name:
                
                df = pd.read_csv(uploaded_file[0])
                st.session_state['user_dataframe'] = df
                st.write(df)
                columns_list=df.columns.tolist()
                col = st.selectbox('Select Column:', columns_list,key="col")
                prompt = "<s>[INST] You are a unstructured data analyzer and you are good at summarizing free text. [/INST] Below are responses from XXX.Each response is seperated by semicolon.(Responses will be inserted here)</s><s>[INST] Write a 200 words paragraph to summarize above received responses based on overall sentiment of the responses. Make sure the summary covers the main points of both positive and negative responses.Do not make recommendations. Do not use bullet points. [/INST]"
                prompt_final = st.text_area("Please review the prompt and modify as needed: ",prompt,key="prompt")
                summary_submit=st.form_submit_button(label="Summary_Submit", on_click=submitted)
                if summary_submit:
                    st.write(st.session_state.submitted)
                

    if st.session_state.submitted==True:
        
        col = st.session_state.col 
        prompt_final = st.session_state.prompt
        df = st.session_state['user_dataframe']
        
        raw_text = ''
        
        for text in df[col]:
            raw_text = raw_text+text+';'
        if prompt_final:
            prompt_final = prompt_final.replace("(Responses will be inserted here)",raw_text)
            payload = {
                "inputs": prompt_final,
                "parameters": {"max_new_tokens": 1000, "do_sample": True,"temprature":0.5}
            }
        if 'endpoint_name' in st.session_state:
            endpoint_name=st.session_state['endpoint_name']
            response = query_endpoint(payload,endpoint_name)
            st.write(response['generated_text'])
        st.session_state.submitted = False

        
    if sentiment_submit and uploaded_file:
        st.session_state.sentiment_submitted = False
        with st.form(key="sentiment_form"):
            def sentiment_submitted():
                st.session_state.sentiment_submitted = True
                st.session_state.col 
                st.session_state.prompt
            if ".pdf" in uploaded_file[0].name:
                raw_text= get_pdf_text(uploaded_file)
            elif ".csv" in uploaded_file[0].name:
                df = pd.read_csv(uploaded_file[0])
                #st.session_state['user_dataframe'] = df
                st.write(df)
                columns_list=df.columns.tolist()
                col = st.selectbox('Select Column:', columns_list,key="col")
                prompt =  """\ <s>[INST]Determine the sentiment of the statement based on the given aspect.
        
        Statement: "I am always able to openly discuss opportunities with my manager as well as explore additional developmental opportunities.  There does not seem to be an appetitie (nor do we include in our hiring strategy) an appetite to allow for longer term assignments requiring 100% commitment which would be developmental. "
        Aspect:management support
        Sentiment: positive
        ###
        Statement: "I am always able to openly discuss opportunities with my manager as well as explore additional developmental opportunities.  There does not seem to be an appetitie (nor do we include in our hiring strategy) an appetite to allow for longer term assignments requiring 100% commitment which would be developmental. "
        Aspect:talent development
        Sentiment: Negative
        ###
        Statement: "Various ad hoc assignments that come about, i.e.: providing coverage for a peer while they may be away."
        Aspect:crosstrain
        Sentiment: Neutral
        ###
        Statement: {}
        Aspect:{}
        Sentiment:[/INST]"""
                prompt_final = st.text_area("Please review the prompt and modify as needed: ",prompt,key="prompt")
                sentiment_submit = st.form_submit_button(label="Sentiment Analysis Submit", on_click=sentiment_submitted)

    if st.session_state.sentiment_submitted == True:
        col = st.session_state.col
        prompt_final = st.session_state.prompt
        df = st.session_state['user_dataframe']
            
        if 'endpoint_name' in st.session_state:
            endpoint_name=st.session_state['endpoint_name']
            client = boto3.client("runtime.sagemaker")
            df["Sentiment_Aspect1"]=df.apply(lambda x: sentiment_analysis(x[col],x["Aspect1"],prompt_final,endpoint_name,client), axis=1)
            df["Sentiment_Aspect1"]=df.apply(lambda x: sentiment_analysis(x[col],x["Aspect1"],prompt_final,endpoint_name,client), axis=1)
        st.session_state.sentiment_submitted = False  
        
    if categrize_submit and uploaded_file:
        st.session_state.categrize_submitted = False
        with st.form(key="categrize_form"):
            def categrize_submitted():
                st.session_state.categrize_submitted = True
                st.session_state.col 
                st.session_state.prompt
            if ".pdf" in uploaded_file[0].name:
                raw_text= get_pdf_text(uploaded_file)
            elif ".csv" in uploaded_file[0].name:
                df = pd.read_csv(uploaded_file[0])
                st.session_state['user_dataframe'] = df
                st.write(df)
                columns_list=df.columns.tolist()
                col = st.selectbox('Select Column:', columns_list,key="col")
                prompt =  """\ <s>[INST]
Please tag below statement into one or more labels using below provided labels. The tagging should be based on the description of each label. Do not tag more than 2 labels per statement.
ONLY inlucde the final labels in the output.Do not output original statement. Do not output explanation. 
    
    label 1.Service 
    label 2.Food quality
    label 3:Food portion
    label 4:price
    label 5:location
    label 6:cleaness

    For example:
    Statement:Food is fresh, but the price is high
    Label: food quality, price
    
    Statement: It's not easy to find parking spot near the resteraunt.
    Label: location
    
    Geenrate the labels following the format in above examples.
    Statement:{}
    Label:[INST]"""
                prompt_final = st.text_area("Please review the prompt and modify as needed: ",prompt,key="prompt")
                cagegrize_submit=st.form_submit_button(label="Categrize Submit", on_click=categrize_submitted)

    if st.session_state.categrize_submitted == True:
        
        st.write("start topic labeling")
        col = st.session_state.col
        prompt_final = st.session_state.prompt
        df = st.session_state['user_dataframe']
        st.write(df)
        if 'endpoint_name' in st.session_state:
            endpoint_name=st.session_state['endpoint_name']
            client = boto3.client("runtime.sagemaker")
            df["Aspects"]=df.apply(lambda x: categrize_analysis(x[col],prompt_final,endpoint_name), axis=1)
            df["Aspects"]=df["Aspects"].apply(lambda x: x.split('\n')[0])
            df["Aspects"]=df["Aspects"].apply(lambda x: x.lower())
            df["Aspects"]=df["Aspects"].apply(lambda x: x.replace("label","").replace(":",""))
            df["Aspect1"]=df["Aspects"].apply(lambda x: x.split(',')[0])
            df["Aspect2"]=df["Aspects"].apply(lambda x: split_func_1(x,','))
            st.write("completed topic labeling")
            st.write(df)
            st.session_state['user_dataframe']=df
        st.session_state.categrize_submitted = False         
        
    if vector_store_submit and uploaded_file:
        st.success(f"File  uploaded. Processing to build vector database...")
        with st.spinner("Building vector database..."):
            raw_text = get_pdf_text(uploaded_file)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            if 'endpoint_name' in st.session_state:
                endpoint_name=st.session_state['endpoint_name']
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,endpoint_name)
            st.success("Vector database built successfully!")
            st.session_state['file_processed'] = vectorstore

    if 'endpoint_name' in st.session_state and 'file_processed' in st.session_state :
        st.write(f"Endpoint '{st.session_state['endpoint_name']}' is ready for questions.")

        st.header("Chat with multiple Docs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()