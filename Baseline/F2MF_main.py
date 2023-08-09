import sys
import time
import numpy as np
import pickle
import tensorflow as tf

from ML_data_processing import item_id_list, user_id_list, train_data, test_data, user_id_list_test, user_id_list_test_male, user_id_list_test_female, test_data_male, test_data_female, df_user_profile  # Uncomment this for Movie lens data processing
#from amazon_data_processing import item_id_list, user_id_list, train_data, test_data, user_id_list_test, user_id_list_test_male, user_id_list_test_female, test_data_male, test_data_female, df_user_profile  # Uncomment this for Amazon data processing


def user_update(single_user_vector, user_rating_list, encrypted_item_vector):
  
    item_vector = encrypted_item_vector.astype('float32')
    #initializing the gradient matrix
    gradient = np.zeros([len(item_vector), len(single_user_vector)]).astype('float32')
    for item_id, rate,_  in user_rating_list:            #(for douben dataset use (for item_id, rate in user_rating_list:))
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = lr * (-2 * error * single_user_vector + 2 * reg_v * item_vector[item_id])
    #print(np.linalg.norm(gradient))
    

    
    return single_user_vector, gradient


def loss():
    loss = []
    
    # User updates
    for i in range(len(user_id_list)):
        for r in range(len(train_data[user_id_list[i]])):    
            item_id, rate,_= train_data[user_id_list[i]][r]        
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)
    return np.mean(loss)

def loss_per_user(user_index, index):
    user_loss = []
    for r in range(len(train_data[user_index])):    
        item_id, rate,_= train_data[user_index][r]       
        error = (rate - np.dot(user_vector[index], item_vector[item_id])) ** 2
        user_loss.append(error)
    user_mean_loss = np.mean(user_loss)
    
    return user_mean_loss

def test_rmse(user_id_list_test, test_data):
    prediction = []
    real_label = []

    # testing
    for i in range(len(user_id_list_test)):
        p = np.dot(user_vector[i:i + 1], np.transpose(item_vector))[0]

        r = test_data[user_id_list_test[i]]

        real_label.append([e[1] for e in r])
        prediction.append([p[e[0]] for e in r])

    max_length = len(max(prediction, key=len))
    prediction = np.array([np.pad(x, (0, max_length - len(x)), mode='constant') for x in prediction])
    
    max_L = len(max(real_label, key=len))
    real_label = np.array([np.pad(x, (0, max_L - len(x)), mode='constant') for x in real_label])
    
    prediction = np.array(prediction, dtype=np.float32)
    real_label = np.array(real_label, dtype=np.float32)
    
    print("prediction",prediction,file=f)
    
    
    print("real_label",real_label,file=f)
    
    print('rmse', np.sqrt(np.mean(np.square(real_label - prediction))),file=f)
    
    return np.sqrt(np.mean(np.square(real_label - prediction)))

if __name__ == '__main__':
    hidden_dim=100
    max_iteration = 10     # [10, 12, 80]
    reg_u = 1e-4
    reg_v = 1e-4
    lr = 0.003
    PHO = 1
    betas = [0.0, 0.3, 0.5, 0.7, 0.9]
    for BETA in betas:
 
            print('########################## BETA = {} ##########################'.format(BETA))
            print('Number of items', len(item_id_list))
            print('Number of users',len(user_id_list))

            #print('Number of training ratings', np.sum([len(train_data[e]) for e in train_data]))
            #print('Number of testing ratings',np.sum([len(test_data[e]) for e in test_data]))
        # Init process
            user_vector = np.zeros([len(user_id_list), hidden_dim]) + 0.01
            user_vector = user_vector.astype('float32')
            item_vector = np.zeros([len(item_id_list), hidden_dim]) + 0.01
            
            
          
            t = time.time()
            
            encrypted_item_vector = item_vector
           
            total_time = []
            training_loss=[]
            costing=[]
            server_time=[]
            n_iterations=[]
            
            A = 1
            B = 1
            for iteration in range(max_iteration):
                #print("A", A)
                #print("B",B)
                print(iteration)
                personal_sum_noise = {uid: np.random.randn(2) *0.05 for uid in user_id_list}
                personal_count_noise = {uid: np.random.randn(2) *0.05 for uid in user_id_list}
            
                delta_A = []
                delta_B = []
                A_count = []
                B_count = []
            

                n_iterations.append(iteration+1)
            

                t = time.time()

                encrypted_gradient_from_user = []
                user_time_list = []
                for i in range(len(user_id_list)):
                    t = time.time()
                    loss_u = loss_per_user(user_id_list[i], i)
                    user_vector[i], gradient = user_update(user_vector[i], train_data[user_id_list[i]], encrypted_item_vector)
                    
                    if df_user_profile.loc[df_user_profile[0] == int(user_id_list[i]), 5].iloc[0] == 'active':             # For Movie-Lens uncomment this
                    #if df_user_profile.loc[df_user_profile[0] == user_id_list[i], 1].iloc[0] == 'active':                # For Amazon Movies uncomment this 
                            C = PHO if A > B else -PHO if A < B else 0
                            scalar = BETA * C * np.power(np.abs(A-B), PHO - 1)
                            D = 1 - scalar
        
                            A_sum = (1 - loss_u) + personal_sum_noise[user_id_list[i]][0]
                            B_sum = 0 + personal_sum_noise[user_id_list[i]][1]
                            A_cnt = 1 + personal_count_noise[user_id_list[i]][0]
                            B_cnt = 0 + personal_count_noise[user_id_list[i]][1]
                    else:
                            C = -PHO if A > B else PHO if A < B else 0
                            scalar = BETA * C * np.power(np.abs(A-B), PHO - 1)
                            D = 1 - scalar
            
                            A_sum = 0 + personal_sum_noise[user_id_list[i]][0]
                            B_sum = (1 - loss_u) + personal_sum_noise[user_id_list[i]][1]
                            A_cnt = 0 + personal_count_noise[user_id_list[i]][0]
                            B_cnt = 1 + personal_count_noise[user_id_list[i]][1]
                    #print("D", D)
                    delta_A.append(A_sum)
                    delta_B.append(B_sum)
                    A_count.append(A_cnt)
                    B_count.append(B_cnt)
                    
                    gradient = D*gradient
                    #print(np.linalg.norm(gradient))
                    gradient_tensor = tf.convert_to_tensor(gradient)
                    gradient_tensor = tf.clip_by_norm(gradient_tensor, 0.01*(1 + BETA))
                    gradient = gradient_tensor.numpy()
                    gradient += np.random.laplace(0,0.0045,size=gradient.shape)                               #  LAMBDA = 0.0045/0.0055
                
                    user_time_list.append(time.time() - t)
                 
                    encrypted_item_vector = encrypted_item_vector - gradient
                
                t = time.time()
                
     
                #print(np.sum(A_count))
                #print(np.sum(B_count))
                A = np.sum(delta_A)/np.sum(A_count)
                B = np.sum(delta_B)/np.sum(B_count)
           
                # for computing loss
                item_vector = np.array([[e for e in vector] for vector in encrypted_item_vector])
                
                print('loss', loss())
                training_loss.append(loss())
                
                if iteration >= (max_iteration - 5):                  
                    total_rmse = test_rmse(user_id_list_test, test_data)
                    rmse_male = test_rmse(user_id_list_test_male, test_data_male)
                    rmse_female = test_rmse(user_id_list_test_female, test_data_female)
                    
                    print("Total RMSE: ", total_rmse)
                    print("RMSE Male: ", rmse_male)
                    print("RMSE Female: ", rmse_female)
                    print("Test Disparity: ", np.abs(rmse_male - rmse_female))
               


