#The New Chat Page as part of the Sports Tweet Bot

#Import the required Libraries
import streamlit as st
import uuid
import json
from datetime import datetime

#Using the AIMessage and HumanMessage to format the Display in the Chat Page
from langchain_core.messages import HumanMessage, AIMessage

#The below used for very rudimentary Summarization of the Chats before Saving.  Needs improvement
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


#Get the Retrieval Chain from the Session State
qaChain = st.session_state.qaChain

#Some funny logic to ensure that the Current Chat ID and the Messages state are reset only if we traverse to the 
#New Chat page from other pages.  
if st.session_state.visitedNewPage == None:
    print("Newpage not visited, so resetting and setting it as visited")
    st.session_state.currentChatID = None
    st.session_state.messages = []
    st.session_state.visitedNewPage = True

#Function to Show the Chat History
def showChatHistory():
    messages = st.session_state.get("messages", [])
    print(f"showChatHistory - Total Messages is: {len(messages)}")
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("BOT"):
                st.markdown(message.content)    

#Start with the Chat
if question := st.chat_input("For Cricket Trivia, Ask Here:"):
    st.session_state.messages.append(HumanMessage(content=question))
    #Accept the Question from the chat_input box and send it to the Retrieval Chain.  
    result = qaChain({"query": question})
    st.session_state.messages.append(AIMessage(content=result['result']))
    showChatHistory()

#The function to use Sumy library to do a summarization to ~50 Characters.  Needs some rework for intelligent Summaries
def generateSummary(text, maxChars=50):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    sentences = 2
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences)
    summaryText = ' '.join([str(sentence) for sentence in summary])
    summaryText = summaryText[:maxChars-3] + '...'
    return summaryText
    #return ' '.join([str(sentence) for sentence in summary])

#Function to Save the Conversation to a File.  Accepts a Bool to determine if it needs to refresh the Chat History Or not
def saveConversations(showHistory):
    messages = st.session_state.get("messages", [])
    if len(messages) == 0:
        st.error("No Messages in the Conversation to Save")
        st.stop()

    #If its a New Chat, no Chat ID exists, so Create one
    if st.session_state.currentChatID == None:
        print("CurrentChatID is None, so Creating a new Chat ID")
        chatID = str(uuid.uuid4())
        st.session_state.currentChatID = chatID
    
    #Create the Blank object for the Chat ID
    st.session_state.chats[st.session_state.currentChatID] = {'messages': [], 'summary': "", 'created_at': datetime.now().isoformat()}
    messageForSummary = ""
    #Loop through the Messages collection and based on Human or AI Message add the right role
    for message in messages:
        if isinstance(message, HumanMessage):
            st.session_state.chats[st.session_state.currentChatID]['messages'].append ({
                    'role': 'user',
                    'content': message.content
                    })
            messageForSummary = messageForSummary + message.content
        elif isinstance(message, AIMessage):
            st.session_state.chats[st.session_state.currentChatID]['messages'].append ({
                    'role': 'bot',
                    'content': message.content
                    })
    
    #Get the Chat Summary
    chatSummary = generateSummary(messageForSummary)
    print(chatSummary)
    st.session_state.chats[st.session_state.currentChatID]['summary'] = chatSummary

    #Finally Save the Conversations to the JSON file
    with open('botconversations.json', 'w') as f:
        json.dump(st.session_state.chats, f)
    if showHistory:    
        showChatHistory()    

#Button Handling for Save Conversation
saveConversation = st.sidebar.button("Save Conversation")
if saveConversation:
    saveConversations(True)

#Button Handling for Clear Conversation
clearButton = st.sidebar.button("Clear Conversation", key="clear")
if clearButton or "messages" not in st.session_state:
    st.sidebar.warning("This will Clear all Conversations and Reset.")
    saveConversations(False)
    st.session_state.currentChatID = None
    st.session_state.messages = []
    #st.session_state.messages = [HumanMessage(content="You are a helpful AI assistant. Reply your answer in markdown format.")]
