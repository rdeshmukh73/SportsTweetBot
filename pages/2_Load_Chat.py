#File to Load the Conversations that are Chosen by the User

#Import the libraries
import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage

#If you've come here, then you are not in the New Chat Page, so reset it
if st.session_state.visitedNewPage:
    print("**Load Chat - Removing the flag for New Page Visited")
    st.session_state.visitedNewPage = None

#Load the Conversations from the JSON file
def loadConversations():
    try:
        with open('botconversations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

#Function to Show the Chat History that is chosen
def showChatHistory():
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("BOT"):
                st.markdown(message.content)   

#Start here for LoadChat
print("Loading Conversations")
st.session_state.messages = []
st.session_state.chats = chats = loadConversations()
if len(st.session_state.chats) == 0:
    st.error("No Conversations available. Create One to Load and View.")
    st.stop()

#Get the Chat IDs list
chatIDs = list(st.session_state.chats.keys())
#Show the Chat IDs along with the Summary and the TimeStamp
selectedChatID = st.selectbox("Select a Chat", chatIDs,
                              format_func=lambda x: f"{st.session_state.chats[x]['summary']} ({st.session_state.chats[x]['created_at']})")
for message in st.session_state.chats[selectedChatID]['messages']:
    role = message['role']
    content = message['content']
    if role == 'user':
        st.session_state.messages.append(HumanMessage(content=content))
    if role == 'bot':
        st.session_state.messages.append(AIMessage(content=content))    
showChatHistory()                
