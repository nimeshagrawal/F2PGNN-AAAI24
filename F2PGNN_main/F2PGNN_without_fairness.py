'''
The base code is adopted from Wu, Chuhan, et al. "A federated graph neural network framework for privacy-preserving personalization." Nature Communications 13.1 (2022): 3091. 
[https://github.com/wuch15/FedPerGNN]

'''
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
#tf.config.set_visible_devices([], 'GPU')


tf.__version__
print(tf.config.list_physical_devices('GPU'))



import pandas as pd
import numpy as np
import random
import h5py
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
import pickle
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
df_user_profile[5] = df_user_profile[0].map(group_dict)
#print(df_user_profile)
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
   # ratings = np.where(all_rating_mat > 0, 1, 0) - train_rating_mat
    return ratings



all_data, user_key, user_value, item_key, item_val, users_id_dict = generate_matrix(df_all)
train_ratings = generate_train_mat(df_train, user_key, user_value, item_key, item_val)
val_ratings = generate_val_mat(df_val, user_key, user_value, item_key, item_val)
test_ratings = generate_test_mat(all_data,train_ratings,val_ratings)


test_ratings.shape


#Hyperparameters 

LABEL_SCALE = 5.0                #LABEL_SCALE is to convert labels in range 0 to 1 (According to rating values for different datasets). 
HIDDEN=64
DROP=0.2
BATCH_SIZE=16
HIS_LEN=20
PSEUDO=1000
NEIGHBOR_LEN=20
CLIP=0.1
LR=0.01
EPS=1
EPOCH=3

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
    history=np.array(history,dtype='int32')
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
    trainlabel=np.array(trainlabel,dtype='float32')
  

     
    return trainu,traini,trainlabel,train_user_index

def generate_valid_data(Ovalid,M):
    #build test user&items
    valu=[]
    vali=[]
    vallabel=[]

    for i in range(Ovalid.shape[0]):
        for j in range(len(Ovalid[i])):
            if Ovalid[i][j]!=0:
                valu.append(i)
                vali.append(j)
                vallabel.append(M[i][j]/LABEL_SCALE)

    valu=np.array(valu,dtype='int32')
    vali=np.array(vali,dtype='int32')
    vallabel=np.array(vallabel,dtype ='float32')
    #testlabel=np.array(testlabel,dtype='int32')
    return valu,vali,vallabel

def generate_test_data(Otest,M, users_id_dict, df_user_profile, gender):
    #build test user&items
    testu=[]
    testi=[]
    testlabel=[]

    for i in range(Otest.shape[0]):
        if(df_user_profile[5][i] == gender):                     #Filter column index as per data and sensitive feature
            for j in range(len(Otest[i])):
                if Otest[i][j]!=0:
                    testu.append(i)
                    testi.append(j)
                    testlabel.append(M[i][j]/LABEL_SCALE)
    
    testu=np.array(testu,dtype='int32')
    testi=np.array(testi,dtype='int32')
    testlabel=np.array(testlabel, dtype = 'float32')
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
    testlabel=np.array(testlabel, dtype = 'float32')
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

def get_model(Otraining,SEED,hidden=HIDDEN,dropout=DROP):
      with tf.device("/gpu:0"):    
        tf.random.set_seed(SEED)
        userembedding_layer = Embedding(Otraining.shape[0]+3, hidden, trainable=True)

        itemembedding_layer = Embedding(Otraining.shape[1]+3, hidden, trainable=True)

        userid_input = Input(shape=(1,), dtype='float32')
        itemid_input = Input(shape=(1,), dtype='float32')

        ui_input = Input(shape=(HIS_LEN,), dtype='float32')
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
        model.compile(loss=['mse'], optimizer='sgd', metrics=['mse'])
       
        return model,userembedding_layer,itemembedding_layer
     

# Private Graph Expansion

def graph_embedding_expansion(Otraining,usernei,local_ciphertext,local_mapping_dict,alluserembs):

    cipher2userid = {}
    for userid,i in enumerate(local_ciphertext):
        for j in i:
            if j not in cipher2userid:
                cipher2userid[j] = [userid]
            else:
                cipher2userid[j].append(userid)

    #third-party server prepares data
                
    send_data = []
    for userid,i in tqdm(enumerate(local_ciphertext)):
        neighbor_info={}
        for j in i:
            neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
            if len(neighbor_id):
                neighbor_info[j] = neighbor_id
        send_data.append(neighbor_info)
        
    #third-party server distributes send_data   
    
    
    #local clients expand graphs
    user_neighbor_emb = []
    for userid,user_items in tqdm(enumerate(usernei)):
        receive_data = send_data[userid]
        decrypted_data = {local_mapping_dict[item_key]:receive_data[item_key] for item_key in receive_data}
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
    


# Batch data for TRAINING

def generate_batch_data_train(batch_size,train_user_index,trainu,traini,history,trainlabel,user_neighbor_emb):
   # idx = np.array(list(train_user_index.keys()))
###############################################################
    idx =list(np.array(list(train_user_index.keys())))
    random.seed(0)
    num_indices = int(0.9*len(idx))
    idx = np.array(list(random.sample(idx, num_indices)))
    print(len(idx))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(idx), batch_size*(i+1)))] for i in range(len(idx)//batch_size+1) if len(range(batch_size*i, min(len(idx), batch_size*(i+1))))]
    print(len(batches))
    while (True):
        for i in batches:
            idxs=[train_user_index[u] for u in i]
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
            
            
            yield ([uid,iid,ui,uneiemb], [y])


#Batch data for TESTING

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




def loss_1(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def custom_loss(model, x, y_true):
   
    with tf.GradientTape() as tape:
        y_pred = model(x, training = True)
        l1 = loss_1(y_true, y_pred)
    return l1,tape.gradient(l1, model.trainable_weights)

optimizer = tf.keras.optimizers.SGD(learning_rate=LR,clipnorm = CLIP)

def test(model,user_neighbor_emb,u,i,label,history):
    testgen=generate_batch_data_test(BATCH_SIZE,u,i,history,label,user_neighbor_emb)
    cr = model.predict_generator(testgen, steps=len(label)//BATCH_SIZE+1,verbose=1)

    #print(cr.flatten()-testlabel)
    #print('rmse:',np.sqrt(np.mean(np.square(cr.flatten()-testlabel/LABEL_SCALE)))*LABEL_SCALE)
    print('rmse_check:',np.sqrt(np.mean(np.square(cr.flatten()-label)))*LABEL_SCALE)

def train(model):
    for rounds in range(EPOCH):
        alluserembs=userembedding_layer.get_weights()[0]
        #user_neighbor_emb=graph_embedding_expansion(Otraining,usernei,alluserembs)
        user_neighbor_emb=graph_embedding_expansion(Otraining,usernei,local_ciphertext,local_mapping_dict,alluserembs)
        traingen=generate_batch_data_train(BATCH_SIZE,train_user_index,trainu,traini,usernei,trainlabel,user_neighbor_emb)
        cnt=0 
        batchloss=[]
        for i in traingen:
            layer_weights=model.get_weights()
           # with tf.device('/CPU:0'):
            loss, grads = custom_loss(model,i[0],i[1])
            #loss=model.train_on_batch(i[0],i[1])
            batchloss.append(loss)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            now_weights=model.get_weights() 
        
            sigma=np.std(now_weights[0]-layer_weights[0])
            #noise of pseudo interacted items
            norm=np.random.normal(0, sigma/np.sqrt(PSEUDO*BATCH_SIZE/now_weights[0].shape[0]), size=now_weights[0].shape) 
           # now_weights[0]+=norm
            itemembedding_layer.set_weights([now_weights[0]])
            print(np.mean(batchloss))
            #ldp noise
            for i in range(len(now_weights)):
                now_weights[i]+=np.random.laplace(0,LR*2*CLIP/np.sqrt(BATCH_SIZE)/EPS,size=now_weights[i].shape)
                #now_weights[i]+=np.random.laplace(0,2*CLIP*EPOCH/EPS,size=now_weights[i].shape)
            model.set_weights(now_weights)
            cnt+=1
            if cnt%10==0:
                print(cnt,loss)
            if cnt==len(train_user_index)//BATCH_SIZE:
                break
        test(model,user_neighbor_emb, valu, vali, vallabel,usernei_val)
    return user_neighbor_emb


 
# Main 

if __name__ == "__main__":
   # os.environ['CUDA_VISIBLE_DEVICES'] = '' 
   # tf.config.set_visible_devices([], 'GPU')
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
#load data
    M = all_data
    Otraining = train_ratings
    Otest = test_ratings
    Ovalid = val_ratings

    print('There are %i interactions logs.'%np.sum(np.array(np.array(M,dtype='bool'),dtype='int32')))
    
    #preprocess data    
    usernei=generate_history(Otraining)
    usernei_test = generate_history(Otest)
    usernei_val = generate_history(Ovalid)
    local_ciphertext = []
    for i in tqdm(usernei):
        messages = []
        for j in i:
            if j!= Otraining.shape[1]+2:
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)

#     local id-ciphertext mapping
    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'):j for j in range(Otraining.shape[1]+3)}
   

    trainu,traini,trainlabel,train_user_index=generate_training_data(Otraining,M)
   # testu,testi,testlabel=generate_test_data(Otest,M)
    valu, vali, vallabel = generate_valid_data(Ovalid,M)

    testu_m,testi_m,testlabel_m = generate_test_data(Otest,M, users_id_dict, df_user_profile,'active')
    testu_f,testi_f,testlabel_f = generate_test_data(Otest,M, users_id_dict, df_user_profile,'inactive')
    testu,testi,testlabel = generate_test_data_all(Otest,M, users_id_dict, df_user_profile)
    #generate public&private keys     
    generate_key()
    
    #build model
    model,userembedding_layer,itemembedding_layer = get_model(Otraining,0)
    #train
    user_neighbor_emb_new = train(model)

    print("Total test loss")
    test(model,user_neighbor_emb_new,testu,testi,testlabel,usernei_test)
    print("Male/Active test loss")
    test(model,user_neighbor_emb_new,testu_m,testi_m,testlabel_m, usernei_test)
    print("Female/Inactive test loss")
    test(model,user_neighbor_emb_new,testu_f,testi_f,testlabel_f, usernei_test)
 

