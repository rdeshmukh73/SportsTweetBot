#Background code to create the Metadata for the Tweet Bot

#Import the libraries

#Using spacy for the Name and Places Entity extraction from the Tweet Data
import spacy
import pandas as pd
import json
from collections import Counter

# Load the pre-trained model
nlp = spacy.load('en_core_web_sm')

#Read the main Tweets File
data = pd.read_csv("E:PathToYourCricketTweetsCSV\\CricketTweets3.csv")

#This helper function will remove any Apostrophe ('s) in the names.
def cleanList(data):
    cleanData = set()
    for item in data:
        if item and not item.endswith("'s"):
            cleanData.add(item)
    return list(cleanData)        

# Remove full matches
def remove_full_matches(places, names):
    return [place for place in places if place not in names]

# Remove partial matches
def remove_partial_matches(places, names):
    def is_partial_match(place, names):
        return any(name in place for name in names)
    
    return [place for place in places if not is_partial_match(place, names)]

#The function to process the Tweets and Extract Names and Places
def processTweets():
    print("Use Spacy Call")
    readTill = '('
    names = []
    places = []
    for index, row in data.iterrows():
        text = row['tweet']
        print(text)
        # Process the text for Names and Places
        doc = nlp(text)
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                #print(ent.text)
                tempName = ent.text.split(readTill)[0]
                #print(tempName)
                #Sometimes tweet text has 2 names together and this will split them. 
                if '&' in tempName:
                    tNames = tempName.split('&', 2)
                    for tName in tNames:
                        names.append(tName)
                else:        
                    names.append(tempName)
            if ent.label_ in ["GPE", "LOC"]:
                tempPlace = ent.text.split(readTill)[0]
                if '&' in tempPlace:
                    tPlaces = tempPlace.split('&', 2)
                    for tPlace in tPlaces:
                        places.append(tPlace)
                else:        
                    places.append(tempPlace)

    #Total Tweets in the Collection
    totalTweets = len(data)

    #Find out the unique Tweet Authors
    uniquePair = data[['name', 'handle']].drop_duplicates()
    pairList = list(uniquePair.itertuples(index=False, name=None))

    nameCounts = Counter(names)
    names = cleanList(names)
    places = cleanList(places)
    for name, count in nameCounts.items():
        print(f"{name}: {count}")

    # Clean the places list to remove any Names wrongly appearing in the Places list
    places_cleaned = remove_full_matches(places, names)
    places_cleaned = remove_partial_matches(places_cleaned, names)

    #Now write this into a JSON
    metaData = {
        "names": names,
        "nameCounts" : nameCounts,
        "places": places_cleaned,
        "featuredAuthors": pairList,
        "totalTweets": totalTweets
    }
    #Write the Metadata into the JSON
    with open("metadata.json", 'w') as json_file:
        json.dump(metaData, json_file, indent=4)


if __name__ == "__main__":
    processTweets()

