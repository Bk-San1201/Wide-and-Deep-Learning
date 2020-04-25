import tensorflow as tf
from tensorflow import feature_column
import numpy as np
from tensorflow.keras import layers, optimizers
from DataLoader import Data
from sklearn import metrics


def get_columns(embed_dim):
    # category columns
    genres_list = ['Sci-Fi', 'Fantasy', "Children's", 'Horror', 'Action', 'Mystery', 'Film-Noir', 'Musical',
                   'Crime', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Romance', 'Documentary', 'Thriller',
                   'Western', 'War']
    genres = feature_column.categorical_column_with_vocabulary_list('genres', genres_list)

    age_list = [1, 18, 25, 35, 45, 50, 56]
    age = feature_column.categorical_column_with_vocabulary_list('age', age_list)
    age_cate = feature_column.indicator_column(age)

    gender = feature_column.categorical_column_with_vocabulary_list('gender', [0, 1])
    gender_cate = feature_column.indicator_column(gender)

    occupation_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    occupation = feature_column.categorical_column_with_vocabulary_list('occupation', occupation_list)
    occupation_cate = feature_column.indicator_column(occupation)

    user = feature_column.categorical_column_with_vocabulary_list('userId', np.arange(1, 6041))
    user_cate = feature_column.indicator_column(user)
    movie = feature_column.categorical_column_with_vocabulary_list('movieId', np.arange(1, 3884))
    movie_cate = feature_column.indicator_column(movie)

    # continuous columns
    year_cont = feature_column.numeric_column('year', dtype=tf.int32)
    age_cont = feature_column.numeric_column('age', dtype=tf.int32)
    gender_cont = feature_column.numeric_column('gender', dtype=tf.int32)

    # wide columns
    crossed_columns = [
        feature_column.crossed_column([genres, age], hash_bucket_size=1000),
        feature_column.crossed_column([genres, occupation], hash_bucket_size=1000),
    ]
    wide_columns = [user_cate, movie_cate, age_cate, gender_cate, occupation_cate,
                    feature_column.indicator_column(crossed_columns[0]),
                    feature_column.indicator_column(crossed_columns[1])]

    # deep columns
    deep_columns = [
        feature_column.embedding_column(user, dimension=embed_dim),
        feature_column.embedding_column(movie, dimension=embed_dim),
        gender_cont,
        year_cont,
        age_cont,
        feature_column.indicator_column(genres)
    ]
    return wide_columns, deep_columns


class Wide_Deep:
    def __init__(self, max_genres, layers=[256, 64, 16, 1], embed_dim=32):
        self.layers = layers
        self.max_genres = max_genres
        self.embed_dim = embed_dim
        self.wide_columns, self.deep_columns = get_columns(self.embed_dim)
        self.model = self.get_model()

    def get_model(self):
        feature_layer_inputs = {}
        # numeric column
        for header in ['userId', 'movieId', 'age', 'year', 'occupation', 'gender']:
            feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header, dtype=tf.int32)
        feature_layer_inputs['genres'] = tf.keras.Input(shape=(self.max_genres,), dtype=tf.string, name='genres')

        wide_layer_feature = layers.DenseFeatures(self.wide_columns, name='wide_layer_feature')
        deep_layer_feature = layers.DenseFeatures(self.deep_columns, name='deep_layer_feature')
        # wide component
        wide_layer = wide_layer_feature(feature_layer_inputs)
        wide_layer = layers.Dense(self.layers[-1])(wide_layer)

        # deep component
        deep_layer = deep_layer_feature(feature_layer_inputs)
        for i in range(len(self.layers)):
            deep_layer = layers.Dense(self.layers[i], activation=tf.nn.relu, kernel_initializer='lecun_uniform',
                                      kernel_regularizer='l2', name='layer_' + str(i))(deep_layer)

        added = layers.Add()([wide_layer, deep_layer])
        last_layer = layers.Dense(units=1, activation='sigmoid')(added)
        last_layer = layers.Lambda(lambda x: x * 4 + 1)(last_layer)
        model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=last_layer)
        for v in feature_layer_inputs.values():
            print(v)
        return model

    def compile(self):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.001),
                           metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])


if __name__ == '__main__':
    wide_deep = Wide_Deep(max_genres=3)

    data = Data('./ml-1m.csv', batch_size=128)
    train_ds, val_ds, test_ds = data.get_data()

    wide_deep.compile()
    print(wide_deep.model.summary())
    wide_deep.model.fit(train_ds, validation_data=val_ds, epochs=2)
    # predict
    y_predict = wide_deep.model.predict(test_ds, use_multiprocessing=True)
    wide_deep.model.evaluate(test_ds)

