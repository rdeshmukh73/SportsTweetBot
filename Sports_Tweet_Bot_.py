#By: Raghavendra Deshmukh (https://github.com/rdeshmukh73)
#Purpose: 
# Collect Interesting Triva Tweets and Threads about Cricket (Players, Matches, Tournaments) and create a ChatBot to answer questions
# Technologies used: LangChain, Mistral AI Chat, Huggingface Embeddings, Streamlit
# App gives ability to Import Tweets, Create New Chat, Save Conversations, Load Saved Conversations
# Strictly for Learning Purposes of RAG, LLM via LangChain
#Inspiration: After doing this short course: https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/1/introduction

#Notes:
#1. There are a lot of print statements which can be removed

#Import the necessary Libraries
import streamlit as st
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from collections import Counter

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


#Setup your Mistral API Key
os.environ["MISTRAL_API_KEY"] = "Your MistralAI API Key"

#Path to Vector DB - We are using ChromaDB which will be created in the below path
chromaDBPath = "crictweets/chroma"

#File where the Cricket related Tweets are Saved
tweetFileName = "E:\\PathToYourCricketCSVFile\\CricketTweets3.csv"
#Mask File used to create the Word Cloud which is in the shape of a Cricket Batsman playing a Shot
maskImage = "E:PathToYourMasKPNGFile\\cricketWordCloud.png"

# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'currentChatID' not in st.session_state:
    st.session_state.currentChatID = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
st.session_state.visitedNewPage = None    #Used to control navigation to the NewChat Page

#Load the Cricket tweet CSV File
def load_csv_file(tweetFileName):
    print(f"LoadCSVFile - Start with {tweetFileName}")
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

# Create Document splits
def create_splits(documents):
    print("CreateSplits - Start")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"CreateSplits - Splits Created {len(splits)}")
    return splits

# Get vector DB Store
# If it exists it will be loaded, if not it will be created in the path "chromaDBPath"
# Using the cache_resource decorator to not have this function run every time
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

# Set up retrieval chain by using the relevant LLM Model and the Prompt Template
# Here we use Mistral AI but users are free to use their relevant LLM Model
# We setup a Prompt Template to be specific to Cricket and not a generic one
@st.cache_resource
def setup_retrieval_chain(_vectorstore):
    print("SetupRetrievalChain - Start")
    #Using the Mistral Small Latest will give decent but not so great answers but will evidently cost lesser tokens
    #llm = ChatMistralAI(model="mistral-small-latest") 


    llm = ChatMistralAI(model="mistral-large-latest")
    print("SetupRetrievalChain - Created LLM with Mistral Small Latest")
    
    # Generic Prompt Template
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer: """
    
    #This Template is specific to Cricket Terminology
    prompt_template = """You are an AI assistant specializing in cricket information based on tweets. Use the following pieces of context from cricket-related tweets to answer the question at the end. Focus on information about cricket players, matches, scores, and events.

    Context from tweets:
    {context}

    When answering, consider the following:
    1. Specific details about cricket players mentioned (e.g., performance, statistics, career highlights)
    2. Information about cricket matches (e.g., teams playing, venue, date, result)
    3. Notable events or moments from the matches
    4. Any cricket-specific terminology or jargon used in the tweets

    If the information isn't explicitly stated in the context, don't speculate. If you don't have enough information to answer the question, simply say that you don't have sufficient details from the given tweets.

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # We setup the Retriever from the Vector Store (ChromaDB) with the search type as similarity. Can use MMR as well.
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})  #, "fetch_k": 20})
    print(f"SetupRetrievalChain - Created Prompt - {PROMPT} and trying to create QA Chain")

    #Create the QAChain which will use the LLM, Retriever and the Prompts to Query based on the Question and return an Answer
    #Note that here I am bringing the source documents which is useful to showcase the source from where the answers were got
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,   #vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("SetupRetrievalChain - End")
    return qa_chain


#This function Reads the metadata.json file and creates the objects required by the Main Page
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


#Function to Create a Word Cloud.  Takes a Dictionary which has Names/Places and their Counts
#The useMask Flag is used only to print the custom Cricket Batter Pic for the names of the Players and not used for the Places/Venues
def generateWordcloud(namesDict, useMask=False):
    # Convert the dictionary to a space-separated string
    text = " ".join(f"{name} " * count for name, count in namesDict.items())

    # Load the mask image
    if useMask == True:
        mask = np.array(Image.open(maskImage))
        wordcloud = WordCloud(width=700, height=350, 
                          background_color='white', 
                          min_font_size=10, mask=mask, contour_width=3, contour_color='steelblue').generate(text)
    else:
        #Creates the Word Cloud for the Places
        wordcloud = WordCloud(width=700, height=550, 
                          background_color='white', 
                          min_font_size=10, contour_width=3, contour_color='steelblue').generate(text)    
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


# Main process
def main():
    #Setup the Main Streamlit Page in a Wide mode
    st.set_page_config(
        page_title="Cricket Tweets Chatbot - By Raghavendra",
        page_icon="🏏",
        layout="wide"
    )

    #Get the Metadata.  Note that I am not yet using the names.
    names, places, nameCounts, featuredAuthors, totalTweets = readMetaData()
    # Create vector store
    vectorstore = get_vector_store()

    # Set up retrieval chain and store it in the Session State to be used in the New Chat Page
    qaChain = setup_retrieval_chain(vectorstore)
    st.session_state.qaChain = qaChain

    st.sidebar.success("Select an Action")
    
    col1, col2 = st.columns(2)
    #Show some Metadata about the App.  No Auto update of the Version (yet)
    with col1:
        st.subheader("App Details")
        st.write("Version: 1.0 Beta")
        st.write(f"Last Updated: 17th Aug 2024")
        st.write("Powered by: MistralAI, LangChain, HuggingFace, Streamlit")
        
        st.subheader("Features")
        st.write("- Intelligent conversations")
        #st.write("- Data analysis capabilities")
        #st.write("- Multi-language support")
    with col2:
        st.subheader("Featured Tweet Authors")
        for name, handle in featuredAuthors:
            st.write(f"**{name}: {handle}**")
        st.write(f"The system has **{totalTweets} Tweets collection**")    

    #Place for the Word Clouds of Players and Places
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Players Mentioned...")
        fig = generateWordcloud(nameCounts, useMask=True)
        st.pyplot(fig)
    with col4:
        st.subheader("Places and Venues...")  
        placesDict = Counter(places)
        fig = generateWordcloud(placesDict, useMask=False)  
        st.pyplot(fig)


# Start here
if __name__ == "__main__":
    main()    
