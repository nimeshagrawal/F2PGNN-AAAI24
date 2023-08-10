'''
Some part of the code is adopted from Wu, Chuhan, et al. "A federated graph neural network framework for privacy-preserving personalization." Nature Communications 13.1 (2022): 3091. 
[https://github.com/wuch15/FedPerGNN]

'''


import tensorflow as tf


tf.__version__
print(tf.config.list_physical_devices('GPU'))



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


import pandas as pd
import numpy as np
import random
import pickle
#import h5py
import os
from scipy.io import savemat
from keras.layers import *
from keras.models import Model
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
#from keras.utils.layer_utils import Layer, InputSpec
from keras import initializers   
from keras.optimizers import *
import keras

from Crypto import Random
from tqdm import tqdm
#from tqdm.notebook import tqdm
import base64
from Crypto.PublicKey import RSA 
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256

#Demo for ML-100K, change the data paths accordingly
current_path =  "./F2PGNN AAAI/Datasets/"

df_all = pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/multicore_data.tsv"), sep = '\t', header = None)
print(df_all)
df_all = df_all.sort_values(by=[0,3])
df_train =  pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/train.tsv"), sep = '\t', header = None)
df_val =  pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/val.tsv"), sep = '\t', header = None)
df_test =  pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/test.tsv"), sep = '\t', header = None)
df_user_profile = pd.read_csv(os.path.join(current_path, "./ML-100K/meta_data/user.meta"), sep = '\t', header = None)


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

# Example usage
active_group, inactive_group, act_dict, avg_count = get_user_activity_info(os.path.join(current_path, "./ML-100K/tsv_data/multicore_data.tsv"))

def get_user_group_info(file_path):
        uA, uI, u_activity_dict, uT = get_user_activity_info(file_path)
        print(f"user activity: {len(uA)}(A) -- {len(uI)}(I), threshold ({uT})")
        group_dict = {uid: 'active' if freq > uT else 'inactive' for uid, freq in u_activity_dict.items()}
        feature_values = ['active', 'inactive']

        return group_dict, feature_values

group_dict, feature_values = get_user_group_info(os.path.join(current_path, "./ML-100K/tsv_data/multicore_data.tsv"))
df_user_profile[1] = df_user_profile[0].map(group_dict)



print(df_user_profile)


def generate_matrix(df):
        user_id = df[0].unique()
        item_id = df[1].unique()
        #print(item_id)
        users = dict(enumerate(user_id))
        key_user_list = list(users.keys())
        #print(key_user_list)
        val_user_list = list(users.values())
        items = dict(enumerate(item_id))
        key_list = list(items.keys())
        #rint(key_list)  
        val_list = list(items.values())
        #rint(val_list)
        #key_list[val_list.index(df[1][4])]


        ratings = np.zeros((len(user_id), len(item_id)))
        #rint(ratings.shape)
        for k in range(len(df)):
            i = key_user_list[val_user_list.index(df[0][k])]
            j = key_list[val_list.index(df[1][k])]
            ratings[i][j] = df[2][k]
        
        return ratings, key_user_list, val_user_list, key_list, val_list, users

def generate_train_mat(df, key_user_list, val_user_list, key_list, val_list):
    ratings = np.zeros((len(key_user_list), len(key_list)))
    for k in range(len(df)):
            i = key_user_list[val_user_list.index(df[0][k])]
            j = key_list[val_list.index(df[1][k])]
            ratings[i][j] = 1
    return ratings

def generate_val_mat(df, key_user_list, val_user_list, key_list, val_list):
    ratings = np.zeros((len(key_user_list), len(key_list)))
    for k in range(len(df)):
            i = key_user_list[val_user_list.index(df[0][k])]
            j = key_list[val_list.index(df[1][k])]
            ratings[i][j] = 1
    return ratings

def generate_test_mat(all_rating_mat, train_rating_mat, val_rating_mat):
    ratings = np.where(all_rating_mat > 0, 1, 0) - train_rating_mat - val_rating_mat
    return ratings


all_data, user_key, user_value, item_key, item_val, users_id_dict = generate_matrix(df_all)
train_ratings = generate_train_mat(df_train, user_key, user_value, item_key, item_val)
val_ratings = generate_val_mat(df_val, user_key, user_value, item_key, item_val)
test_ratings = generate_test_mat(all_data,train_ratings, val_ratings)


#Hyperparamters


LABEL_SCALE = 5.0       #LABEL_SCALE is to convert labels in range 0 to 1 (According to rating values for different datasets). 
HIDDEN=64
DROP=0.2
BATCH_DROP = 0.1
BATCH_SIZE = 256
HIS_LEN=20
NEIGHBOR_LEN=20
LR=0.05
EPOCH= 50
ALPHA = 1

# Generating required training, validation and test data along with user history

def generate_history(Otraining):
    #build user history
    history=[]
    for i in range(Otraining.shape[0]):
        user_history=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0.0:
                user_history.append(j)
        random.seed(0)
        random.shuffle(user_history)
        user_history=user_history[:HIS_LEN] 
        history.append(user_history+[Otraining.shape[1]+2]*(HIS_LEN-len(user_history)))
    history = np.array(history,dtype='int32')
    return history

def generate_training_data(Otraining,M):
    #build training user&items
    trainu=[]
    traini=[]
    trainlabel=[]
    train_user_index={}
    for i in range(Otraining.shape[0]):
        user_index=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0:
                user_index.append(len(trainu))
                trainu.append(i)
                traini.append(j)
                trainlabel.append(M[i][j]/LABEL_SCALE)
        if len(user_index):
            
            train_user_index[i]=user_index
    

    trainu=np.array(trainu,dtype='int32')
    traini=np.array(traini,dtype='int32')
    trainlabel=np.array(trainlabel,dtype = 'float32')
   
    
    return trainu,traini,trainlabel,train_user_index


def generate_validation_data(Ovalid,M):
    #build training user&items
    valu=[]
    vali=[]
    vallabel=[]
    val_user_index={}
    for i in range(Ovalid.shape[0]):
        user_index=[]
        for j in range(len(Ovalid[i])):
            if Ovalid[i][j]!=0:
                user_index.append(len(valu))
                valu.append(i)
                vali.append(j)
                vallabel.append(M[i][j]/LABEL_SCALE)
        if len(user_index):
            
            val_user_index[i]=user_index
    

    valu=np.array(valu,dtype='int32')
    vali=np.array(vali,dtype='int32')
    vallabel=np.array(vallabel,dtype = 'float32')
    
    return valu,vali,vallabel,val_user_index

def generate_test_data(Otest,M, users_id_dict, df_user_profile, activity):
   
    testu=[]
    testi=[]
    testlabel=[]

    for i in range(Otest.shape[0]):
        if(df_user_profile[1][i] == activity):                          #Filter column index as per data and sensitive feature
            for j in range(len(Otest[i])):
                if Otest[i][j]!=0:
                    testu.append(i)
                    testi.append(j)
                    testlabel.append(M[i][j]/LABEL_SCALE)
    
    testu=np.array(testu,dtype='int32')
    testi=np.array(testi,dtype='int32')
    testlabel=np.array(testlabel,dtype = 'float32')
    #testlabel=np.array(testlabel,dtype='int32')
    return testu,testi,testlabel

def generate_test_data_all(Otest,M, users_id_dict, df_user_profile):
    #build test user&items
    testu=[]
    testi=[]
    testlabel=[]

    for i in range(Otest.shape[0]):
        for j in range(len(Otest[i])):
            if Otest[i][j]!=0:
                testu.append(i)
                testi.append(j)
                testlabel.append(M[i][j]/LABEL_SCALE)
    
    testu=np.array(testu,dtype='int32')
    testi=np.array(testi,dtype='int32')
    testlabel=np.array(testlabel,dtype = 'float32')
    #testlabel=np.array(testlabel,dtype='int32')
    return testu,testi,testlabel

# Encryption of Item IDs

def generate_key():
    random_generator = Random.new().read
    rsa = RSA.generate(1024, random_generator)
    public_key = rsa.publickey().exportKey()
    private_key = rsa.exportKey()
    
    with open('rsa_private_key.pem', 'wb')as f:
        f.write(private_key)
        
    with open('rsa_public_key.pem', 'wb')as f:
        f.write(public_key)
    

def get_key(key_file):
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key    

def sign(msg):
    private_key = get_key('rsa_private_key.pem')
    signer = PKCS1_signature.new(private_key)
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))
    return signer.sign(digest)

def verify(msg, signature):
    #use signature because the rsa encryption lib adds salt defaultly
    pub_key = get_key('rsa_public_key.pem')
    signer = PKCS1_signature.new(pub_key)
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))
    return signer.verify(digest, signature)
    
def encrypt_data(msg): 
    pub_key = get_key('rsa_public_key.pem')
    cipher =encryptor = PKCS1_OAEP.new(pub_key)
    encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf8"))))
    return encrypt_text.decode('utf-8')

def decrypt_data(encrypt_msg): 
    private_key = get_key('rsa_private_key.pem')
    cipher = PKCS1_OAEP.new(private_key)
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg))
    return back_text.decode('utf-8')


# Model definition

class ComputeMasking(keras.layers.Layer):
    def __init__(self, maskvalue=0,**kwargs):
        self.maskvalue=maskvalue
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, self.maskvalue)
        return K.cast(mask, K.floatx())*(-99)

    def compute_output_shape(self, input_shape):
        return input_shape
    

def get_model(Otraining, SEED, hidden=HIDDEN,dropout=DROP):
#     import tensorflow
#     tensorflow.random.set_seed(2)
    with tf.device('/device:GPU:0'):
        tf.random.set_seed(SEED)
        userembedding_layer = Embedding(Otraining.shape[0]+3, hidden, trainable=True)
        #userembedding_layer = Embedding(Otraining.shape[0], hidden, trainable=True)
        itemembedding_layer = Embedding(Otraining.shape[1]+3, hidden, trainable=True)
        #itemembedding_layer = Embedding(Otraining.shape[1], hidden, trainable=True)

        userid_input = Input(shape=(1,), dtype='int32')
        itemid_input = Input(shape=(1,), dtype='int32')

        ui_input = Input(shape=(HIS_LEN,), dtype='int32')
        neighbor_embedding_input = Input(shape=(HIS_LEN,NEIGHBOR_LEN,hidden), dtype='float32')
        mask_neighbor = Lambda(lambda x:K.cast(K.cast(K.sum(x,axis=-1),'bool'),'float32'))(neighbor_embedding_input)

        neighbor_embeddings = TimeDistributed(TimeDistributed(Dense(hidden)))(neighbor_embedding_input)

        uiemb = Dense(hidden,activation='sigmoid')(itemembedding_layer(ui_input))
        uiembrepeat = Lambda(lambda x :K.repeat_elements(K.expand_dims(x,axis=2),NEIGHBOR_LEN,axis=2))(uiemb) 
        attention_gat = Reshape((HIS_LEN,NEIGHBOR_LEN))(LeakyReLU()(TimeDistributed(TimeDistributed(Dense(1)))(concatenate([uiembrepeat,neighbor_embeddings]))))
        attention_gat = Lambda(lambda x:x[0]+(1-x[1])*(-99))([attention_gat,mask_neighbor])
        agg_neighbor_embeddings = Lambda(lambda x:K.sum(K.repeat_elements(K.expand_dims(x[0],axis=3),hidden,axis=3)*x[1],axis=-2))([attention_gat,neighbor_embeddings])

        uiemb_agg = Dense(hidden)(concatenate([agg_neighbor_embeddings,uiemb]))  
        uemb = Dense(hidden,activation='sigmoid')(Flatten()(userembedding_layer(userid_input)))
        uemb = Dropout(dropout)(uemb)
        iemb = Dense(hidden,activation='sigmoid')(Flatten()(itemembedding_layer(itemid_input)))
        iemb = Dropout(dropout)(iemb)

        masker = ComputeMasking(Otraining.shape[1]+2)(ui_input)
        uembrepeat = Lambda(lambda x :K.repeat_elements(K.expand_dims(x,axis=1),HIS_LEN,axis=1))(uemb) 

        attention = Flatten()(LeakyReLU()(Dense(1)(concatenate([uembrepeat,uiemb_agg]))))
        attention = add([attention,masker])
        attention_weight = Activation('softmax')(attention)
        uemb_g = Dot((1, 1))([uiemb, attention_weight])
        uemb_g = Dense(hidden)(concatenate([uemb_g, uemb]))

        out = Dense(1,activation='sigmoid')(concatenate([uemb_g, iemb]))
        model = Model([userid_input,itemid_input,ui_input,neighbor_embedding_input],out)
       
        
        return model,userembedding_layer,itemembedding_layer
     
# Private Graph Expansion

def graph_embedding_expansion(Otraining,usernei,local_ciphertext,local_mapping_dict,alluserembs):

    cipher2userid = {}                                   
    for userid,i in tqdm(enumerate(local_ciphertext)):
        for j in i:
            if j not in cipher2userid:
                cipher2userid[j] = [userid]
            else:
                cipher2userid[j].append(userid)

    #third-party server prepares data
    
    ############ send data is a list of dictionaries where each dictionary contains item id(as messages) which
    ############ each user has interacted with as keys and value is list of embeddings of all users which have interacted 
    ############ with this item id (the key)
    #print("Step 3")  
    send_data = []
    for userid,i in tqdm(enumerate(local_ciphertext)):
        neighbor_info={}
        for j in i:
            neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
            #print(neighbor_id)
            if len(neighbor_id):
                neighbor_info[j] = neighbor_id
        send_data.append(neighbor_info)
    
    #third-party server distributes send_data   
    
    #print("Step 4")
    #local clients expand graphs

    '''
    receive data is a dictionary with key as messages and values as list of embeddings
    decrypted_data is a dictionary with key as item ids and values as list of embeddings
   
    For each item, neighbor_embs store list of embeddings of 100 neighbors because Neighbor_len is 100
    Hence array size is 100*64
    
    all_neighbor_embs stores embeddings for each item. Hence size is 100*100*64 as there are 100 items in user history
    because His_len is 100.
    
    user_neighbor_emb is for all users hence shape is 943*100*100*64

    '''

    user_neighbor_emb = []
    for userid,user_items in enumerate(usernei):
        receive_data = send_data[userid]     
        decrypted_data = {local_mapping_dict[item_key]:receive_data[item_key] for item_key in receive_data}  
        
        #print("Decrypted Data:", decrypted_data)
        all_neighbor_embs=[]
        for item  in user_items:
            if item in decrypted_data:
                neighbor_embs = decrypted_data[item]
                random.shuffle(neighbor_embs)
                neighbor_embs = neighbor_embs[:NEIGHBOR_LEN] 
                neighbor_embs += [[0.]*HIDDEN]*(NEIGHBOR_LEN-len(neighbor_embs))
                
            else:
                neighbor_embs = [[0.]*HIDDEN]*NEIGHBOR_LEN
            all_neighbor_embs.append(neighbor_embs)
            
        all_neighbor_embs = all_neighbor_embs[:HIS_LEN]
        all_neighbor_embs += [[[0.]*HIDDEN]*HIS_LEN]*(HIS_LEN-len(all_neighbor_embs))
        user_neighbor_emb.append(all_neighbor_embs)
           
    user_neighbor_emb = np.array(user_neighbor_emb,dtype='float32')
    return user_neighbor_emb
    


def generate_batch_data_train_and_valid(train_user_index,trainu,traini,history,trainlabel,user_neighbor_emb):
   
    idx = list(np.array(list(train_user_index.keys())))
   
    random.seed(0)

    num_indices = int((1 - BATCH_DROP)*len(idx))
   

    batches = list(random.sample(idx, num_indices))

    output_dict = {}
    for user in batches:
   
        idxs=[train_user_index[user]]

        uid=np.array([])
        iid=np.array([])
        uneiemb=user_neighbor_emb[:0]
        y=np.array([])
        for idss in idxs:
            uid=np.concatenate([uid,trainu[idss]])
            iid=np.concatenate([iid,traini[idss]])
            y=np.concatenate([y,trainlabel[idss]])
            uneiemb=np.concatenate([uneiemb,user_neighbor_emb[trainu[idss]]],axis=0)
        uid=np.array(uid,dtype='int32')
        iid=np.array(iid,dtype='int32')
        ui=history[uid]
        uid=np.expand_dims(uid,axis=1)
        iid=np.expand_dims(iid,axis=1)

        output_dict[user] = ([uid,iid,ui,uneiemb], [y])

    yield (output_dict, batches)



def generate_batch_data_test(batch_size,testu,testi,history,testlabel,user_neighbor_emb):
    idx = np.arange(len(testlabel))
    np.random.shuffle(idx)
    y=testlabel
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            uid=np.expand_dims(testu[i],axis=1)
            iid=np.expand_dims(testi[i],axis=1)
            ui=history[testu[i]]
            uneiemb=user_neighbor_emb[testu[i]]
     
            yield ([uid,iid,ui,uneiemb], [y])


# F2PGNN : Customized functions


def loss_1(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def loss_2(A,B,regularisation_factor, ALPHA):
    return tf.constant(regularisation_factor) * float(tf.pow(tf.math.abs(A-B), ALPHA))

def custom_loss(model, x, y_true, A,B,regularisation_factor, ALPHA):
   
    with tf.GradientTape() as tape:
        y_pred = model(x, training = True)
        l1 = loss_1(y_true, y_pred)
        l2 = loss_2(A,B,regularisation_factor, ALPHA)
        loss = tf.math.add(l1, l2)
    return l1, l2, loss, tape.gradient(l1, model.trainable_weights)


def calculate_array_norms(array_list):
    norms = []
    for array in array_list:
        norm = np.linalg.norm(array)
        norms.append(norm)
    return norms

optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

############################################################################
def test(model,user_neighbor_emb,testu, testi, testlabel, history):
    #print(user_neighbor_emb)
    #testgen=generate_batch_data(BATCH_SIZE,testu,testi,usernei,testlabel,user_neighbor_emb)
    testgen=generate_batch_data_test(BATCH_SIZE,testu,testi,history,testlabel,user_neighbor_emb)
    cr = model.predict_generator(testgen, steps=len(testlabel)//BATCH_SIZE+1,verbose=1)

   
    print('rmse_check:',np.sqrt(np.mean(np.square(cr.flatten()-testlabel)))*LABEL_SCALE)
    
    return np.sqrt(np.mean(np.square(cr.flatten()-testlabel)))*LABEL_SCALE

def valid(model, epoch, P_valid, Q_valid, lamda,user_neighbor_emb):
    valgen=generate_batch_data_train_and_valid(val_user_index,valu,vali,usernei_val,vallabel,user_neighbor_emb)
    for j in valgen:
        P_per_val = []
        Q_per_val = []
        P_add_val = []
        Q_add_val = []
        batch_loss_val = []
        #print(j[1])
        for u in j[1]:   
            y_pred_val = model(j[0][u][0], training = False)
            l1 = loss_1(j[0][u][1], y_pred_val)
            l2 = loss_2(P_valid,Q_valid,lamda, ALPHA)
            loss_u_val = tf.math.add(l1, l2)
            batch_loss_val.append(l1)
            
            if df_user_profile[1][u] == 'active':
                    P_per = (1 - l1.numpy())
                    Q_per = 0 
                    P_add = 1 
                    Q_add = 0 

            else:         
                    P_per = 0 
                    Q_per = (1 - l1.numpy())
                    P_add = 0
                    Q_add = 1 
            P_per_val.append(P_per)
            Q_per_val.append(Q_per)
            P_add_val.append(P_add)
            Q_add_val.append(Q_add)
        
        P_valid = np.sum(P_per_val)/np.sum(P_add_val)
        Q_valid = np.sum(Q_per_val)/np.sum(Q_add_val)

    return P_valid, Q_valid, np.mean(batch_loss_val)




def train(model,CLIP, L_NOISE, BETA, SIGMA):
    print("############################BETA = {}################".format(BETA))
    P = 1.0
    Q = 1.0
    
    P_valid = 1
    Q_valid = 1
    overall_loss = []
    overall_loss_val = []
    best_val_loss = 100
    var_name = 'disparity_{}_{}_{}_new_amazon'.format(BETA, CLIP, L_NOISE)
    #var_name_user1 = 'P_per_{}_{}_new_amazon'.format(BETA, SIGMA)
    #var_name_user2 = 'Q_per_{}_{}_new_amazon'.format(BETA, SIGMA)
    #var_name_user3 = 'P_add_list_{}_{}_new_amazon'.format(BETA, SIGMA)
    #var_name_user4 = 'Q_add_list_{}_{}_new_amazon'.format(BETA, SIGMA)
    var_name_2 = 'Validation_loss_{}_{}_{}_new_amazon'.format(BETA, CLIP, L_NOISE)
    var_name_3 = 'Test_RMSE_{}_{}_{}_new_amazon'.format(BETA, CLIP, L_NOISE)
    locals()[var_name] = []
    #locals()[var_name_user1] = []
    #locals()[var_name_user2] = []
    #locals()[var_name_user3] = []
    #locals()[var_name_user4] = []
    locals()[var_name_2] = []
    locals()[var_name_3] = []
    cnt=1
#     seed_list = np.arange(0,EPOCH)
    for rounds in range(EPOCH):
        print("Epoch:", cnt)
        alluserembs=userembedding_layer.get_weights()[0]
        #print("alluseremb", alluserembs)
        user_neighbor_emb=graph_embedding_expansion(Otraining,usernei,local_ciphertext,local_mapping_dict,alluserembs)
        #user_neighbor_emb=graph_embedding_expansion(Otraining,usernei,alluserembs)
        #print(user_neighbor_emb.shape)
        traingen=generate_batch_data_train_and_valid(train_user_index,trainu,traini,usernei,trainlabel,user_neighbor_emb)
         
      
        for j in traingen:
#             print(j[1])
            personal_sum_noise = {uid: np.random.randn(2) *SIGMA for uid in j[1]}
            personal_count_noise = {uid: np.random.randn(2) *SIGMA for uid in j[1]}
            batchloss=[]
         

            P_per_list = []
            Q_per_list = []
            P_add_list = []
            Q_add_list = []
            for u in j[1]:   
 
                loss_u, fair_loss_u, loss_total_u, grads_u = custom_loss(model, j[0][u][0], j[0][u][1], P,Q,BETA,ALPHA)
            
                batchloss.append(loss_u)
        
                #Calculating L using L = 1 - BETA*R*|P - Q|^(ALPHA -1)  where R = ALPHA*(-1)^(P<Q)*(-1)^(u not belonging to S0)
                if df_user_profile[1][u] == 'active':
                        R = ALPHA if P > Q else -ALPHA if P < Q else 0
                        scalar = BETA * R * np.power(np.abs(P-Q), ALPHA - 1)
                        L = tf.constant(1 - scalar)
                        
                        P_per = (1 - loss_u.numpy()) + personal_sum_noise[u][0] 
                        Q_per = 0 + personal_sum_noise[u][1] 
                        P_add = 1 + personal_count_noise[u][0] 
                        Q_add = 0 + personal_count_noise[u][1]
                else:         
                        R = -ALPHA if P > Q else ALPHA if P < Q else 0
                        scalar = BETA * R * np.power(np.abs(P-Q), ALPHA - 1)
                        L = tf.constant(1 - scalar)
                    
                        P_per = 0 + personal_sum_noise[u][0] 
                        Q_per = (1 - loss_u.numpy()) + personal_sum_noise[u][1] 
                        P_add = 0 + personal_count_noise[u][0] 
                        Q_add = 1 + personal_count_noise[u][1] 
            
                P_per_list.append(P_per)
                Q_per_list.append(Q_per)
                P_add_list.append(P_add)
                Q_add_list.append(Q_add)
                for i in range(len(grads_u)):
                    grads_u[i] =  tf.convert_to_tensor(grads_u[i])
                
           
                for i in range(len(grads_u)):
                    grads_u[i] = tf.math.scalar_mul(float(L), grads_u[i])
                for i in range(len(grads_u)):
                    grads_u[i] = tf.clip_by_norm(grads_u[i], CLIP*(1 + BETA))
              
                EPS = (2*CLIP*(1 + BETA))/L_NOISE
#################################################################################################################                
                for i in range(len(grads_u)):
                    grads_u[i]+=np.random.laplace(0,2*(1+BETA)*CLIP*LR/EPS,size=grads_u[i].shape)
               
                optimizer.apply_gradients(zip(grads_u, model.trainable_weights))
                now_weights=model.get_weights()
        
                itemembedding_layer.set_weights([now_weights[0]]) 

                model.set_weights(now_weights)

        P = np.sum(P_per_list)/np.sum(P_add_list)
        Q = np.sum(Q_per_list)/np.sum(Q_add_list)

        print(np.sum(P_add_list))
        print(np.sum(Q_add_list))


        overall_loss.append(np.mean(batchloss))
        print("Train loss:", np.mean(overall_loss))
        
        
        
        ##################Evaluate model###############################
        
        P_valid, Q_valid, batch_loss_val = valid(model, rounds, P_valid, Q_valid, BETA, user_neighbor_emb)
  
        print("Val disparity: ", np.abs(P_valid - Q_valid))
        overall_loss_val.append(batch_loss_val)
        #if np.mean(overall_loss_val) < best_val_loss:
        #if batch_loss_val < best_val_loss:
          # best_val_loss = np.mean(overall_loss_val)
        #   best_val_loss = batch_loss_val
        #   disp = np.abs(P_valid - Q_valid)
        #   locals()[var_name].append(disp)
        #else:
        #   locals()[var_name].append(disp)
        print("Val loss:", np.mean(overall_loss_val))
        locals()[var_name].append(np.abs(P_valid - Q_valid))
        #locals()[var_name_user1].append(P_per_list[0]) 
        #locals()[var_name_user2].append(Q_per_list[0])
        #locals()[var_name_user3].append(P_add_list[0])
        #locals()[var_name_user4].append(Q_add_list[0])
        locals()[var_name_2].append(batch_loss_val)
       
        cnt+=1
         ######################## RMSE testing for last 5 iterations (best is considered)  ########################
        test(model,user_neighbor_emb,valu,vali,vallabel,usernei_val)
        if cnt >= (EPOCH - 5):
           print("######Testing for count {}######".format(cnt))
           print("Total test RMSE")
           rmse =  test(model,user_neighbor_emb,testu,testi,testlabel,usernei_test)
           locals()[var_name_3].append(rmse)
           print("Male/Active test RMSE")
           test(model,user_neighbor_emb,testu_a,testi_a,testlabel_a, usernei_test)
           print("Female/Inactive test RMSE")
           test(model,user_neighbor_emb,testu_in,testi_in,testlabel_in, usernei_test)
        
    print(locals()[var_name])

    #Create the folder with name results in the working directory to save the results 
    
    np.save('./results/{}.npy'.format(var_name), locals()[var_name])
    np.save('./results/{}.npy'.format(var_name_2), locals()[var_name_2])
    np.save('./results/{}.npy'.format(var_name_3), locals()[var_name_3])
   # if BETA == 0.3:
       # np.save('./results/{}.npy'.format(var_name_user1), locals()[var_name_user1])
       # np.save('./results/{}.npy'.format(var_name_user2), locals()[var_name_user2])
       # np.save('./results/{}.npy'.format(var_name_user3), locals()[var_name_user3])
       # np.save('./results/{}.npy'.format(var_name_user4), locals()[var_name_user4])
    return user_neighbor_emb

# Main
    
if __name__ == "__main__":
    
    #load data

    M = all_data
    Otraining = train_ratings
    Ovalid = val_ratings
    Otest = test_ratings
    print(Otraining.shape[0], Otraining.shape[1])
    

#########For loading cyphertexts############ 

#Provide the pre-encrypted IDs path

    # with open("cipher_amazon_new_train_20", "rb") as fp:                 
    #     local_ciphertext = pickle.load(fp)
    # with open("cipher_dict_amazon_new_train_20", "rb") as fp:   
    #     local_mapping_dict = pickle.load(fp)

    print('There are %i interactions logs.'%np.sum(np.array(np.array(M,dtype='bool'),dtype='int32')))
    

    #preprocess data    
    usernei = generate_history(Otraining)
    usernei_val = generate_history(Ovalid)
    usernei_test = generate_history(Otest)
    print(usernei)
    print(usernei_val)

    #generate public&private keys     
    generate_key()
    
    local_ciphertext = []
    for i in tqdm(usernei):
        messages = []
        for j in i:
            if j!= Otraining.shape[1]+2:
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)
    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'):j for j in range(Otraining.shape[1]+3)}
    ###########################################
    
    trainu,traini,trainlabel,train_user_index=generate_training_data(Otraining,M)
    valu,vali,vallabel,val_user_index=generate_validation_data(Ovalid,M)
    ####### Parametric details #################
    # trainu : List of userids with number of each ids corresponding to number of rated items by a user in training set
    # traini : List of item ids of each user which he has interacted with.  
    # train_user_index : Dictionary with key being user id and values being list with last entry of each list denoting 
    #                    count ratings given by each user.

    # For Gender as Sensitive Feature

    #testu_m,testi_m,testlabel_m=generate_test_data(Otest,M, users_id_dict, df_user_profile,'M')
    #testu_f,testi_f,testlabel_f=generate_test_data(Otest,M, users_id_dict, df_user_profile,'F')
    #testu,testi,testlabel=generate_test_data_all(Otest,M, users_id_dict, df_user_profile)

    # For Activity as Sensitive Feature

    testu_a,testi_a,testlabel_a=generate_test_data(Otest,M, users_id_dict, df_user_profile,'active')
    testu_in,testi_in,testlabel_in=generate_test_data(Otest,M, users_id_dict, df_user_profile,'inactive')
    testu,testi,testlabel=generate_test_data_all(Otest,M, users_id_dict, df_user_profile)

    clips = [0.2]     # Choose from set {0.2, 0.3, 0.4}
    l_noises = [0.15, 0.20, 0.25, 0.30, 0.35]
    BETAS = [0.5]     # Choose from set {0.3, 0.5, 0.7, 0.9}

    sigmas = [0.005]
    for SIGMA in sigmas:
       for CLIP in clips:
           for L_NOISE in l_noises:
               for BETA in BETAS:
                    model,userembedding_layer,itemembedding_layer = get_model(Otraining,0)

                    user_neighbor_emb_new = train(model,CLIP,L_NOISE, BETA, SIGMA)
      


