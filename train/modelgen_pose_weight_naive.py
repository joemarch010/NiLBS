"""




Builds and then saves an *incredibly* naive weight prediction network using keras.
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

        pose = sample['pose']

        for i in range(0, sample['rest_vertices'].shape[0]):

            vertex = sample['vertices'][i]
            weights = sample['weights'][i]
            query = pose.get_nilbs_encoding(vertex)

            X.append(query)
            Y.append(weights)

    X = np.array(X)
    Y = np.array(Y)

    query_size = X.shape[1]
    n_layer_weights = 80 * n_weights

    model = keras.Sequential([
        layers.Dense(n_layer_weights, input_shape=[query_size], activation='relu'),
        layers.Dense(n_layer_weights, activation='relu'),
        layers.Dense(n_layer_weights, activation='relu'),
        layers.Dense(n_weights, activation='softmax')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    model.summary()

    model.fit(X, Y, epochs=10)

    model.save(model_output_path)

