import tensorflow as tf  # Keras 2.1.2 and TF-GPU 1.8.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random
import multiprocessing
import time

dense_layers = [0]
layer_sizes = [512]
conv_layers = [1]

def create_model_and_train(dense_layer, layer_size, conv_layer):
    def check_data():
        choices = {"no_attacks": no_attacks,
                   "attack_closest_to_hatch": attack_closest_to_hatch,
                   "attack_enemy_structures": attack_enemy_structures,
                   "attack_enemy_start": attack_enemy_start
                   }

        total_data = 0

        lengths = []

        for choice in choices:
            print("Length of {} is: {}\n".format(choice, len(choices[choice])))
            total_data += len(choices[choice])
            lengths.append(len(choices[choice]))

        print("Total data length now is:\n", total_data)
        return lengths
    
    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
    conv_layer, layer_size, dense_layer, int(time.time()))
    print(NAME)

    train_data_dir = "./train_data"


    cwd = os.getcwd()
    print(cwd)

    all_files = os.listdir(train_data_dir)

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same',
                                input_shape=(176,200,3),
                                activation='relu'))

    for l in range(conv_layer-1):
        model.add(Conv2D(layer_size, (3, 3), padding='same',
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

    model.add(Flatten())
    for _ in range(dense_layer):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(0.5))


    model.add(Dense(4, activation='softmax'))

    learning_rate = 0.0001
    opt = Adam(lr=learning_rate, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    #filepath = "./1x512-CNN.model"

    # if you want to load in a previously trained model
    # that you want to further train:
    #tf.keras.models.load_model(filepath)

    hm_epochs = 50

    for i in range(hm_epochs):
        print("Current Epoch: {}/{}".format(i,hm_epochs))
        current = 0
        increment = 200
        not_maximum = True

        
        maximum = len(all_files)
        random.shuffle(all_files)

        while not_maximum:
            print("WORKING ON {}:{}".format(current, current+increment))
            no_attacks = []
            attack_closest_to_hatch = []
            attack_enemy_structures = []
            attack_enemy_start = []

            for file in all_files[current:current+increment]:
                full_path = os.path.join(train_data_dir, file)
                data = np.load(full_path)

                data = list(data)
                for d in data:
                    choice = np.argmax(d[0])
                    if choice == 0:
                        no_attacks.append([d[0], d[1]])
                    elif choice == 1:
                        attack_closest_to_hatch.append([d[0], d[1]])
                    elif choice == 2:
                        attack_enemy_structures.append([d[0], d[1]])
                    elif choice == 3:
                        attack_enemy_start.append([d[0], d[1]])

            lengths = check_data()

            lowest_data = min(lengths)

            random.shuffle(no_attacks)
            random.shuffle(attack_enemy_structures)
            random.shuffle(attack_enemy_start)
            random.shuffle(attack_closest_to_hatch)

            no_attacks = no_attacks[:lowest_data]
            attack_closest_to_hatch = attack_closest_to_hatch[:lowest_data]
            attack_enemy_structures = attack_enemy_structures[:lowest_data]
            attack_enemy_start = attack_enemy_start[:lowest_data]

            check_data()

            train_data = no_attacks + attack_closest_to_hatch + attack_enemy_structures + attack_enemy_start
            random.shuffle(train_data)
            print("Total Training Data: {}".format(len(train_data)))

            test_size = 100
            batch_size = 128

            x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1,176,200,3)
            y_train = np.array([i[0] for i in train_data[:-test_size]])

            x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
            y_test = np.array([i[0] for i in train_data[-test_size:]])

            model.fit(x_train,y_train,
                    batch_size=batch_size,
                    validation_data=(x_test,y_test),
                    shuffle=True,
                    verbose=1, callbacks=[tensorboard])

            #model.save(NAME)
            model.save('1x512-CNN.model')
            current += increment
            if current > maximum:
                not_maximum = False

if __name__ == '__main__':
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                p = multiprocessing.Process(target=create_model_and_train, args=(
                    dense_layer, layer_size, conv_layer))
                p.start()
                p.join()
