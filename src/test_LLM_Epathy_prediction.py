import csv
import pandas as pd
import argparse
import codecs

import torch
from empathy_classifier import EmpathyClassifier
import os
import codecs


'''
Example:
'''

parser = argparse.ArgumentParser("Test")
parser.add_argument("--input_path", type=str, help="path to input test data")
parser.add_argument("--output_path", type=str, help="output file path")

parser.add_argument("--ER_model_path", type=str, help="path to ER model")
parser.add_argument("--IP_model_path", type=str, help="path to IP model")
parser.add_argument("--EX_model_path", type=str, help="path to EX model")

args = parser.parse_args()

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


input_df = pd.read_csv(args.input_path, header=0)

# ids = input_df.id.astype(str).tolist()
# seeker_posts = input_df.seeker_post.astype(str).tolist()
# response_posts = input_df.response_post.astype(str).tolist()

seeker_posts = input_df.text.astype(str).tolist()
column_name = [col for col in input_df.columns if col.startswith('cleaned')][0]
response_posts = input_df[column_name].astype(str).tolist()

empathy_classifier = EmpathyClassifier(device,
						ER_model_path = args.ER_model_path, 
						IP_model_path = args.IP_model_path,
						EX_model_path = args.EX_model_path,)


# Ensure parent directory exists
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
# output_file = codecs.open(args.output_path, 'w', 'utf-8')
output_file = open(args.output_path, 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

# csv_writer.writerow(['id','seeker_post','response_post','ER_label','IP_label','EX_label', 'ER_rationale', 'IP_rationale', 'EX_rationale'])
csv_writer.writerow(['text','author','type','age','gender','race','sentiment','empathy_er','empathy_ip','empathy_ex','num_replies','community','prediction_depression','prediction_suicide',column_name,'ER_label','IP_label','EX_label', 'ER_rationale', 'IP_rationale', 'EX_rationale'])

for i in range(len(seeker_posts)):
    (logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX, predictions_rationale_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])

    csv_writer.writerow([
        seeker_posts[i],
        input_df['author'].iloc[i],
        input_df['type'].iloc[i],
        input_df['age'].iloc[i],
        input_df['gender'].iloc[i],
        input_df['race'].iloc[i],
        input_df['sentiment'].iloc[i],
        input_df['empathy_er'].iloc[i],
        input_df['empathy_ip'].iloc[i],
        input_df['empathy_ex'].iloc[i],
        input_df['num_replies'].iloc[i],
        input_df['community'].iloc[i],
        input_df['prediction_depression'].iloc[i],
        input_df['prediction_suicide'].iloc[i],
        response_posts[i],
        predictions_ER[0],
        predictions_IP[0],
        predictions_EX[0],
        predictions_rationale_ER[0].tolist(),
        predictions_rationale_IP[0].tolist(),
        predictions_rationale_EX[0].tolist()
    ])
# csv_writer.writerow([seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()])

output_file.close()
