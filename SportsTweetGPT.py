import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import uuid

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


os.environ["MISTRAL_API_KEY"] = "KrT9Tk5bsmlSj9NuGCjKRpVQVeRIbWO7"

chromaDBPath = "crictweets/chroma"
csvDirectory = "E:\\Deshmukh-2024\\Learning\\python\\MonthlyExpense"
tweetFileName = "E:\\Deshmukh-2024\\Learning\\python\\cricketanalyzer\\CricketTweets2.csv"
maskImage = "E:\\Deshmukh-2024\\Learning\\python\\cricketanalyzer\\cricketWordCloud.png"

#Load the Cricket tweet CSV File
def load_csv_file(tweetFileName):
    print("LoadCSVFile - Start")
    documents = []
    loader = CSVLoader(tweetFileName, 
                       csv_args={
                           "delimiter": ",",
                           "quotechar": '"',
                           "fieldnames" : ["handle", "name", "tweetdate", "tweet", "isretweet", "likes", "retweets", "comments", "isthread"]
                       }
                    )
    documents.extend(loader.load())
    print(f"LoadCSVFile - There are {len(documents)} Files Loaded")
    return documents

# Create splits
def create_splits(documents):
    print("CreateSplits - Start")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"CreateSplits - Splits Created {len(splits)}")
    return splits

# Get vector DB Store
@st.cache_resource
def get_vector_store():
    print("GetVectorStore - Start")
    # Specify the Hugging Face model to use for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(chromaDBPath):
        print("GetVectorStore - Vector DB Exists and Will be Loaded")
        chromaDB = Chroma(persist_directory=chromaDBPath, embedding_function=embeddings)
        return chromaDB
    
    print("GetVectorStore - Creating New Vector DB")
    documents = load_csv_file(tweetFileName=tweetFileName)
    splits = create_splits(documents)
    chromaDB = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=chromaDBPath)
    chromaDB.persist()
    print("GetVectorStore - Created ChromaDB and Exiting")
    return chromaDB

# Set up retrieval chain
@st.cache_resource
def setup_retrieval_chain(_vectorstore):
    print("SetupRetrievalChain - Start")
    llm = ChatMistralAI(model="mistral-small-latest")
    print("SetupRetrievalChain - Created LLM with Mistral Small Latest")
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})  #, "fetch_k": 20})
    print(f"SetupRetrievalChain - Created Prompt - {PROMPT} and trying to create QA Chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,   #vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("SetupRetrievalChain - End")
    return qa_chain

# Initialize the Messages
def initMessages():
    clearButton = st.sidebar.button("Clear Conversation", key="clear")
    if clearButton or "messages" not in st.session_state:
        st.session_state.messages = [HumanMessage(content="You are a helpful AI assistant. Reply your answer in markdown format.")]


# Load the Dataframe
def readMetaData():
    # Read the JSON file
    with open('metadata.json', 'r') as json_file:
        data = json.load(json_file)
    # Access the lists
    names = data['names']
    places = data['places']
    nameCounts = data['nameCounts']
    featuredAuthors = data['featuredAuthors']
    totalTweets = data['totalTweets']
    return names, places, nameCounts, featuredAuthors, totalTweets


def generateWordcloud(namesDict):
    # Load the mask image
    mask = np.array(Image.open(maskImage))

    # Convert the dictionary to a space-separated string
    text = " ".join(f"{name} " * count for name, count in namesDict.items())
    
    wordcloud = WordCloud(width=700, height=350, 
                          background_color='white', 
                          min_font_size=10, mask=mask, contour_width=3, contour_color='steelblue').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

@st.cache_resource
def mainPage(featuredAuthors, totalTweets, nameCounts):
    st.title("Cricket Tweets Chatbot - By Raghavendra")
    col1, col2 = st.columns(2)

    with col1:
        # Get the current date
        #currentDate = datetime.now().date()
        #formattedDate = currentDate.strftime("%d-%B-%Y")
        st.subheader("App Details")
        st.write("Version: 0.3 Beta")
        st.write(f"Last Updated: 15th Aug 2024")
        st.write("Powered by: MistralAI, LangChain, HuggingFace")
        
        st.subheader("Features")
        st.write("- Intelligent conversations")
        #st.write("- Data analysis capabilities")
        #st.write("- Multi-language support")
    with col2:
        st.subheader("Featured Tweet Authors")
        for name, handle in featuredAuthors:
            st.write(f"**{name}: {handle}**")
        st.write(f"The system has **{totalTweets} Tweets collection**")    

    st.subheader("Players covered...")
    fig = generateWordcloud(nameCounts)
    st.pyplot(fig)

def saveConversation():
    chatID = str(uuid.uuid4())
    st.session_state.chats[chatID] = {'messages': [], 'created_at': datetime.now().isoformat()}
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            st.session_state.chats[chatID]['messages'].append ({
                'role': 'user',
                'content': message.content
            })
        elif isinstance(message, AIMessage):
            st.session_state.chats[chatID]['messages'].append ({
                'role': 'bot',
                'content': message.content
            })
    with open('botconversations.json', 'w') as f:
        json.dump(st.session_state.chats, f)
    return

#Define the Sidebar
def sideBar():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Home", "Chatbot"])

        if st.button("Save Conversation"):
            saveConversation()

    if page == "Chatbot":
        initMessages()    
    return page

#Define the ChatBot Page
def chatPage(qaChain):
    if question := st.chat_input("Ask your Question:"):
        st.session_state.messages.append(HumanMessage(content=question))
        result = qaChain({"query": question})
        st.session_state.messages.append(AIMessage(content=result['result']))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("BOT"):
                st.markdown(message.content)    
    return


# Main process
def main():
    
    st.set_page_config(
        page_title="Cricket Tweets Chatbot - By Raghavendra",
        page_icon="🏏",
        layout="wide"
    )
    names, places, nameCounts, featuredAuthors, totalTweets = readMetaData()
    # Create vector store
    vectorstore = get_vector_store()
    # Set up retrieval chain
    qaChain = setup_retrieval_chain(vectorstore)
   
    page = sideBar()
    if page == "Home":
        mainPage(featuredAuthors, totalTweets, nameCounts)
    if page =="Chatbot":
        chatPage(qaChain)    
   
# Start here
if __name__ == "__main__":
    main()    



#Unused Code
# Setup the Main Page
def setupChatPage(featuredAuthors):
    st.set_page_config(
        page_title="Cricket Tweets Chatbot - By Raghavendra",
        page_icon="🏏",
        layout="wide"
    )
    #st.title("Sports Tweets Chatbot")
    st.header("Cricket Tweets Chatbot")
    st.sidebar.title("Cricket Tweet Chatbot")

    # Display the unique pairs in the sidebar
    st.sidebar.header("Featured Tweet Authors")
    for name, handle in featuredAuthors:
        st.sidebar.write(f"{name}: {handle}")
