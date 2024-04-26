import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# I think for cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from src.motion_refiner_4D import Motion_refiner, MAX_NUM_OBJS
from src.config import *
from src.functions import *
from src.simple_TF_continuos import *

correct_model_path = './models/refined_refined_TF&num_layers_enc:2&num_layers_dec:4&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:64&dense_n:512&num_dense:3&concat_emb:True&features_n:777&optimizer:RMSprop&norm_layer:True&activation:tanh.h5'
correct_model_path = './models/orignial_bert_clip_continous/TF&num_layers_enc:1&num_layers_dec:6&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:32&dense_n:512&num_dense:3&concat_emb:True&features_n:793&optimizer:adam&norm_layer:True&activation:tanh.h5'
if os.path.exists(correct_model_path):
    print('*****path found')
else:
    print('**** path not found')


model_path = models_folder
mr = Motion_refiner(load_models=True ,traj_n = traj_n, locality_factor=True, clip_only=False)
feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = mr.get_indices()
embedding_indices = mr.embedding_indices


#============================== load dataset ==========================================
X,Y, data = mr.load_dataset("latte_100k_lf", filter_data = True, base_path=data_folder)
X_train, X_test, X_valid, y_train, y_test, y_valid, indices_train, indices_test, indices_val = mr.split_dataset(X, Y, test_size=0.2, val_size=0.1)

test_dataset = tf.data.Dataset.from_tensor_slices((mr.prepare_x(X_test),
                                                    list_to_wp_seq(y_test,d=4),
                                                    X_test[:,embedding_indices])).batch(X_test.shape[0])

g = generator(test_dataset,stop=True,augment=False)
x_t, y_t = next(g)

def prepare_x(x):
  objs = list_to_wp_seq(x[:,obj_poses_indices])
  trajs = list_to_wp_seq(x[:,traj_indices])
  return np.concatenate([objs,trajs],axis = 1)


def generator(data_set,stop=False,augment=True, num_objs = 3):

    while True:
        for x, y,emb in data_set:
            x_new, y_new = x,y
            if augment:
                x_new, y_new = augment_xy(x,y,width_shift_range=0.3, height_shift_range=0.3,rotation_range=np.pi,
                        zoom_range=[0.6,1.1],horizontal_flip=True, vertical_flip=True, offset=[-0.5,-0.5])
            else:
                x_new, y_new = augment_xy(x,y,width_shift_range=0.0, height_shift_range=0.0,rotation_range=0.0,
                        zoom_range=0.0,horizontal_flip=False, vertical_flip=False, offset=[-0.5,-0.5])
            # emb[:,-num_batches:] = tf.one_hot(tf.argmax(emb[:,-num_batches:],1),num_objs).numpy()
            # emb_new = tf.concat([emb[:,:-num_batches],tf.one_hot(tf.argmax(emb[:,-num_batches:],1),num_objs)],-1)

            yield ( [x_new , y_new[:, :-1],emb] , y_new[:, 1:] )
        if stop:
            break

def increase_dataset(x,y,embedding_indices,augment_factor):
    x_, y_ = prepare_x(x), list_to_wp_seq(y)
    emb = x[:,embedding_indices]

    x_new = x_
    y_new = y_
    emb_new=emb
    for i in range(augment_factor):
        x_new_i, y_new_i = augment_xy(x_,y_,width_shift_range=0.5, height_shift_range=0.5,rotation_range=np.pi,
                        zoom_range=[0.5,1.5],horizontal_flip=True, vertical_flip=True, offset=[-0.5,-0.5])
        x_new = np.append(x_new,x_new_i, axis=0)
        y_new = np.append(y_new,y_new_i, axis=0)
        emb_new = np.append(emb_new,emb, axis=0)

    print("new data shape: x=",x_new.shape,"   y=",y_new.shape,"   emb=", emb_new.shape)
    return x_new, y_new, emb_new

def evaluate_model(model, epoch):

    print("epoch:",epoch)
    print("\nwith data augmentation:")
    # result_eval_aug = model.evaluate(generator(test_dataset,stop=True))[0]
    x_test_new, y_test_new, emb_test_new= increase_dataset(X_test ,y_test,embedding_indices,10)
    result_eval_aug = model.evaluate((x_test_new, y_test_new[:,:-1,:], emb_test_new), y_test_new[:,1:,:])[0]


    print("without data augmentation:")
    result_eval = model.evaluate(generator(test_dataset,stop=True, augment=False))[0]
    x_test_new, y_test_new, emb_test_new= increase_dataset(X_test ,y_test,embedding_indices,0)
    result_eval = model.evaluate((x_test_new, y_test_new[:,:-1,:], emb_test_new), y_test_new[:,1:,:])[0]

    print("\n ----------------------------------------")
    print("withdata generation:")


    test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test),
                                                    list_to_wp_seq(y_test),
                                                    X_test[:,embedding_indices])).batch(X_test.shape[0])

    g = generator(test_dataset,stop=True,augment=True)
    x_t, y_t = next(g)
    pred = generate(model ,x_t, traj_n = len(y_t[0,:,0]) + 1).numpy()
    np.save('genAug.npy', pred)
    result_gen_aug = np.average((y_t - pred[:,1:,:])**2)
    print("Test loss w generation and augmentation: ",result_gen_aug)


    g = generator(test_dataset,stop=True,augment=False)
    x_t, y_t = next(g)
    pred = generate(model ,x_t, traj_n = len(y_t[0,:,0]) + 1).numpy()
    np.save('genNoAug.npy', pred)
    result_gen = np.average((y_t - pred[:,1:,:])**2)
    print("Test loss w generation: ",result_gen)


    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    with file_writer.as_default():
        tf.summary.scalar('test_result_gen', data=result_gen, step=epoch)
        tf.summary.scalar('test_result_gen_aug', data=result_gen_aug, step=epoch)
        tf.summary.scalar('test_result_eval', data=result_eval, step=epoch)
        tf.summary.scalar('test_result_eval_aug', data=result_eval_aug, step=epoch)


model_names = ["TF&num_layers_enc:1&num_layers_dec:6&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:32&dense_n:512&num_dense:3&concat_emb:True&features_n:793&optimizer:adam&norm_layer:True&activation:tanh.h5"]

models_metrics = {}

for model_name in model_names:

    # model_file = model_path+model_name
    model_file = correct_model_path
    if os.path.exists(model_file):
        print("model found")
    param = file_name2dict(model_file)
    print("======================================================================================================")
    print("num Encoders: ",param["num_layers_enc"],"\tnum Decoders: ",param["num_layers_dec"],"\tDepth: ",param['d_model'])
    model_tag = "enc:"+str(param["num_layers_enc"])+"-dec:"+str(param["num_layers_dec"])+"-d"+str(param['d_model'])

    model = load_model(model_file)
    metrics = evaluate_model(model, 68)
    models_metrics[model_tag] = metrics
