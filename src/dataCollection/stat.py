import pymongo
from dotenv import load_dotenv
import os
import argparse


def get_number_of_letters_from_database(letter):
    """get the number of letters in the database"""
    ans = 0
    for i in collection.find({"word": letter}):
        ans += 1
    return ans


def delete_letter(letter, times):
    """delete a letter times from the end of the database"""
    letters = list(collection.find({"word": letter}))
    for i in range(times):
        collection.delete_one({"_id": letters[-i - 1]["_id"]})


# connect to databse
load_dotenv()
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client["signspeak"]
collection = db[("data_collection")]
print("connected to database")

# cli arg parser
parser = argparse.ArgumentParser(description="get character and word count")
parser.add_argument("-l", dest="letter", type=str, required=True)
parser.add_argument("-count", dest="count", type=int, required=True)
args = parser.parse_args()


if args.count:
    delete_letter(args.letter, args.times)

d = {}

for i in collection.find():
    if i["word"] in d:
        d[i["word"]] += 1
    else:
        d[i["word"]] = 1

print(sum(d.values()))
print(d)
