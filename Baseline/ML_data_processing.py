import os
import csv
import numpy as np
import itertools
import random
import pandas as pd

def load_csv(fileName, fileWithHeader=False):
    with open(fileName, 'r') as f:
        reader = csv.reader(f, delimiter = '	')
        if fileWithHeader:
            header = next(reader)
        else:
            header = []
        data = [r for r in reader]
    return header, data

def find_index(lst, element):
    for index, tpl in enumerate(lst):
        if tpl[0] == element:
            return index
    return -1
def find_index_lst(lst, element):
    for sublist in lst:
        if sublist[1] == element:
            return sublist
    return None


def get_user_activity_info(file_path):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='	', header=None, names=['UserID', 'ItemID', 'Ratings', 'Timestamp'])
    df = df.sort_values(by=['UserID','Timestamp'])
    # Get the count of ratings for each user
    act_dict = df['UserID'].value_counts().to_dict()
    # Calculate the average count
    avg_count = sum(act_dict.values()) / len(act_dict)

    # Separate active and inactive groups
    active_group = [uid for uid, count in act_dict.items() if count > avg_count]
    inactive_group = [uid for uid, count in act_dict.items() if count <= avg_count]

    return active_group, inactive_group, act_dict, avg_count


def get_user_group_info(file_path):
        uA, uI, u_activity_dict, uT = get_user_activity_info(file_path)
        print(f"user activity: {len(uA)}(A) -- {len(uI)}(I), threshold ({uT})")
        group_dict = {uid: 'active' if freq > uT else 'inactive' for uid, freq in u_activity_dict.items()}
        feature_values = ['active', 'inactive']

        return group_dict, feature_values



#### Ml-100k #####
num_items = 937
num_users = 917

##### ML-1M #####
# num_items = 3043
# num_users = 6022

gender_col_index = 2   # Choose 1 for ML-1M and 2 for ML-100K

predict_step = 3
least_rating_num = 5

current_path =  "./F2PGNN AAAI/Datasets/"

#For ML-100K

headers_train, ratings_train = load_csv(os.path.join(current_path, 'ML-100K/tsv_data/train.tsv'))
headers_test, ratings_test = load_csv(os.path.join(current_path, 'ML-100K/tsv_data/test.tsv'))
df_user_profile = pd.read_csv(os.path.join(current_path, "./ML-100K/meta_data/user.meta"), sep = '	', header = None)
group_dict, feature_values = get_user_group_info(os.path.join(current_path, './ML-100K/tsv_data/multicore_data.tsv'))

#For ML-1M

# headers_train, ratings_train = load_csv(os.path.join(current_path, 'ML-1M/tsv_data/train.tsv'))
# headers_test, ratings_test = load_csv(os.path.join(current_path, 'ML-1M/tsv_data/test.tsv'))
# df_user_profile = pd.read_csv(os.path.join(current_path, "./ML-1M/meta_data/user.meta"), sep = '	', header = None)
# group_dict, feature_values = get_user_group_info(os.path.join(current_path, './ML-1M/tsv_data/multicore_data.tsv'))


df_user_profile[5] = df_user_profile[0].map(group_dict)

#print(ratings_train)
print(df_user_profile)

item_frequent_dict_train= {}

for e in ratings_train:
    item_frequent_dict_train[e[1]] = item_frequent_dict_train.get(e[1], 0) + 1

item_frequent_dict_train = sorted(item_frequent_dict_train.items(), key=lambda x:x[1], reverse=True)
item_id_list = [int(e[0]) for e in item_frequent_dict_train[:num_items]]

user_id_list = sorted(set([e[0] for e in ratings_train]), key=lambda x:int(x))[:num_users]
random.seed(10)
random.shuffle(user_id_list)

 #Client sanpling according to dropout-rate
 
user_id_list = user_id_list[:825]                         #For ML-100K       
# user_id_list = user_id_list[:3011]                      #For ML-1M


print(len(user_id_list))
ratings_dict_train = {e:[] for e in user_id_list}

counter = 0
for record in ratings_train:
    
    if record[0] not in user_id_list or int(record[1]) not in item_id_list:
        continue
    counter += 1
    ratings_dict_train[record[0]].append([item_id_list.index(int(record[1])), float(record[2]), int(record[3])])


train_data = {}

for user_id in ratings_dict_train:
 
    #sorting  according to timestamp
    sorted_rate = sorted(ratings_dict_train[user_id], key=lambda x:x[-1], reverse=False)
    
    train_data[user_id] = sorted_rate[:]

#######################Filtering train data ######################
for item in train_data:
    train_data[item] = train_data[item][:20]                   #Value of m (number of items interacted) is 20

print(len(train_data))
##############################Testing#####################

item_frequent_dict_test= {}

for e in ratings_test:
    item_frequent_dict_test[e[1]] = item_frequent_dict_test.get(e[1], 0) + 1


item_frequent_dict_test = sorted(item_frequent_dict_test.items(), key=lambda x:x[1], reverse=True)
#print(item_frequent_dict_test)
item_id_list_test = [int(e[0]) for e in item_frequent_dict_test[:len(item_frequent_dict_test)]]

user_id_list_test = sorted(set([e[0] for e in ratings_test]), key=lambda x:int(x))[:num_users]

random.seed(10)
random.shuffle(user_id_list_test)


user_id_list_test = user_id_list_test[:825]               #For ML-100K         
# user_id_list_test = user_id_list_test[:3011]            #For ML-1M 

###### Filtering users according to demographic info ###################

user_id_list_test_male = []
user_id_list_test_female = []

#print(user_id_list_test)
for i in user_id_list_test:
    #if df_user_profile.loc[df_user_profile[0] == int(i), gender_col_index].iloc[0] == 'M':                #Uncomment this for Gender as sensitive feature
    if df_user_profile.loc[df_user_profile[0] == int(i), 5].iloc[0] == 'active':
        user_id_list_test_male.append(i)
    else:
        user_id_list_test_female.append(i)
        
ratings_dict_test = {e:[] for e in user_id_list_test}

counter = 0
for record in ratings_test:
    
    if record[0] not in user_id_list_test or int(record[1]) not in item_id_list_test:
        continue
    counter += 1
    ratings_dict_test[record[0]].append([item_id_list_test.index(int(record[1])), float(record[2]), int(record[3])])


test_data = {}

for user_id in ratings_dict_test:
 
    #sorting  according to timestamp
    sorted_rate = sorted(ratings_dict_test[user_id], key=lambda x:x[-1], reverse=False)
    
    test_data[user_id] = sorted_rate[:]
    
    #test_data[user_id] = sorted_rate[-3:]

########################## MALE/ACTIVE  ####################################
ratings_dict_test_male = {e:[] for e in user_id_list_test_male}

counter = 0
for record in ratings_test:
    
    if record[0] not in user_id_list_test_male or int(record[1]) not in item_id_list_test:
        continue
    counter += 1
    ratings_dict_test_male[record[0]].append([item_id_list_test.index(int(record[1])), float(record[2]), int(record[3])])


test_data_male = {}

for user_id in ratings_dict_test_male:
 
    #sorting  according to timestamp
    sorted_rate = sorted(ratings_dict_test_male[user_id], key=lambda x:x[-1], reverse=False)
    
    test_data_male[user_id] = sorted_rate[:]
    

######################  Female/Inactive ############################
ratings_dict_test_female = {e:[] for e in user_id_list_test_female}

counter = 0
for record in ratings_test:
    
    if record[0] not in user_id_list_test_female or int(record[1]) not in item_id_list_test:
        continue
    counter += 1
    ratings_dict_test_female[record[0]].append([item_id_list_test.index(int(record[1])), float(record[2]), int(record[3])])


test_data_female = {}

for user_id in ratings_dict_test_female:
 
    #sorting  according to timestamp
    sorted_rate = sorted(ratings_dict_test_female[user_id], key=lambda x:x[-1], reverse=False)
    
    test_data_female[user_id] = sorted_rate[:]
    

print(len(test_data))
print(len(test_data_male))
print(len(test_data_female))
