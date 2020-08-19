"""




Builds and then saves an *incredibly* naive weight prediction network.
The work uses the naive pose encoding for queries, i.e queries must have dimension 3 + 16 * B.
Only use this model for testing the API.




"""
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


if __name__ == '__main__':

    train_data_path = '../data/weight/weight_small_train.npz'
    model_output_path = '../models/weight_naive'

    train_data = np.load(train_data_path, allow_pickle=True)['arr_0']
    n_weights = train_data[0]['weights'].shape[1]

    X = []
    Y = []

    for sample in train_data:

        pose_encoding = sample['pose'].get_naive_encoding()

        for i in range(0, sample['vertices'].shape[0]):

            vertex = sample['vertices'][i]
            weights = sample['weights'][i]
            query = np.concatenate((vertex, pose_encoding), axis=0)

            X.append(query)
            Y.append(weights)

    X = np.array(X)
    Y = np.array(Y)

    query_size = X.shape[1]

    model = keras.Sequential([
        layers.Dense(query_size, activation='relu', input_shape=[query_size]),
        layers.Dense(query_size, activation='relu'),
        layers.Dense(n_weights),
        layers.Softmax()
    ])


    model.compile(loss='mse', metrics=['mae', 'mse'])

    model.summary()

    model.fit(X, Y)

    model.save(model_output_path)

