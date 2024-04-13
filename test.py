from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="get character and word count")
parser.add_argument('-d', dest='description', type=str, required=True)
args = parser.parse_args()

type_of_model = "Encoder"

for split_number in range(5):
  run_name = type_of_model + "_" + args.description + "_" + str(split_number+1) + "-fold_" + datetime.now().strftime("%m/%d/%Y %H:%M:%S")
  print(run_name)
