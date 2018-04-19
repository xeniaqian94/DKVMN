import pandas as pd 
from collections import defaultdict

df = pd.read_csv("../../data/duolingo/es_en.slam.20171218.train.csv")

user_history = df[['token','part_of_speech','correctness','user']].groupby(['user']).agg({'token':lambda x: tuple(x), 
                                   'part_of_speech':lambda x: tuple(x),'correctness':lambda x:tuple(x)})

directory = ""
pos = "part_of_speech"
token = "token"

skill2id = {}

#user to id mapping
user_list=list(df['user'].unique())
user2id=defaultdict(lambda:len(user2id))
with open(directory+"user_id_mapping_pos_token.txt","w") as f:
    for user in user_list:
        f.write(user+"\t"+str(user2id[user])+"\n")
        

with open(directory+pos+"_"+token+"_train.csv","w") as f:
    for idx, row in user_history.iterrows():
        total = len(row['part_of_speech'])
        pos_tags = row['part_of_speech']
        tokens = row['token']
        skills_list = []
        for i in range(total):
            key = pos_tags[i] + "_" + tokens[i]
            if key in skill2id:
                skills_list.append(skill2id[key])
            else:
                skill2id[key] = str(len(skill2id))
                skills_list.append(skill2id[key])
        
        correctness_list=[str(int(value)) for value in list(row['correctness'])]
        f.write(str(len(skills_list))+" # "+str(user2id[idx])+"\n"+str(",".join(skills_list))+"\n"+",".join(correctness_list)+"\n")
        
print(directory+pos+"_"+token+"_id_mapping.txt")
with open(directory+pos+"_"+token+"_id_mapping.txt","w") as f:
    for skill in skill2id:
        f.write(skill+"\t"+str(skill2id[skill])+"\n")