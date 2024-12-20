import pandas as pd
import os
from tqdm import tqdm

dataroot = 'data/medical_cxr_vqa'
dataset_file = 'medical-cxr-vqa-questions.csv'

# read the dataset
dataset = pd.read_csv(os.path.join(dataroot, dataset_file))


total_answer = set()
for i in tqdm(range(len(dataset))):
    question = dataset.iloc[i]['question']

    if 'what abnormalities are seen in this image' in question:
        answer = dataset.iloc[i]['answer']
        answers = answer.split(',')
        for ans in answers:
            total_answer.add(ans.strip())

print(len(total_answer))