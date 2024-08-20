# SportsTweetBot
A Bot built on Streamlit, MistralAI, Langchain and HuggingFace to process Cricket Tweets and Chat.

1. This is not for commercial purpose and is only meant for my learning of the above mentioned technology.
2. I used python 3.12.5 on my Windows 10 Laptop for this activity.
3. The imported libraries are mentioned in the requirements.txt. Please run it before.
4. Scrape Twitter manually or otherwise to bring the Cricket or Sports Tweets based on the sample format CricketTweets3.csv.
5. Before running the Bot, please run the createMetadata_.py script to Load the Data, Cleanse, Identify Names and Places and Create the metadata.json file.
6. python run createMetadata_.py.
7. Then run the Sports_Tweet_Bot_.py with the command streamlit run Sports_Tweet_Bot_.py.
8. Note that the chat_input and chat_message calls in Streamlit do not work in versions below 1.13.
9. The script takes a while to run on the first instance as it creates the ChromaDB.
10. A sample version of the botconversations.json is provided in the github to get started.
