import random

from pynoahdb.symbolseries import SymbolSeries

PARAM1_WINDOW_SIZES = [3,5,8,13,21,34,55,89,144,233]
PARAM2_WINDOW_SIZES = [5,8,13,21,34,55,89,144,233,377]

PARAM1_RATIO = [0.10,0.25,0.5,0.75,0.90]

OUTER_VARS = ["C({0})","H({0})","L({0})","O({0})","V({0})"]

FUNCTIONS = [
    ["RATIO_TO_RAVG({0},{1})",["t","w"],False],
    ["RATIO_TO_RLINEAR({0},{1})",["t","w"],False],
    ["QUANTILE_RATIO({0},{1},0.25,0.75)",["t","w"],False],
    ["QUANTILE_RATIO({0},{1},0.10,0.90)",["t","w"],False],
    ["MACDFAST({0})",["t"],False],
    ["MACDSLOW({0})",["t"],False],
    ["RSI({0},{1})",["t","w"],False],
    ["STOCH_OSC({0},{1})",["s","w"],True],
    ["RENTROPY({0},{1})",["t","w"],False],
    ["RKURT({0},{1})",["t","w"],False],
    ["RSKEW({0},{1})",["t","w"],False],
    ["RAVG_MEDIAN({0},{1})",["t","w"],False],
    ["RAVGS_RATIO({0},{1},{2})",["t","w","w"],False],
    ["RSEMS_RATIO({0},{1},{2})",["t","w","w"],False],
    ["STOCH_OSC_RAVG({0},{1},{2})",["s","w","w"],True],
    ["W_VOL_AVG({0},{1},{2})",["s","w","w"],True],
]

def generate_functions(f_count):
    #Get the symbols
    symbols = SymbolSeries().symbol_list()

    ret_list = []

    selected_functions = 0

    while selected_functions < f_count:
        #Get the function
        function = random.choice(FUNCTIONS)

        #Get the symbol
        symbol = random.choice(symbols)

        ##if the function requires a symbol and the symbol has only close data
        if function[2] and symbol[1]:
            continue #skip this iteration

        #Get the parameters
        params = []
        for p in function[1]:
            if p == "s":
                params.append(symbol[0])
            elif p == "w":
                params.append(random.choice(PARAM1_WINDOW_SIZES))
            elif p == "t":
                if symbol[1]:
                    params.append(f"C({symbol[0]})")
                else:
                    metric = random.choice(OUTER_VARS)
                    params.append(metric.format(symbol[0]))

        #Add the function to the list
        ret_list.append(function[0].format(*params))

        selected_functions += 1

    return "|".join(ret_list)
                
# from pynoahdb.series import Series
from pyfunc.processor import Processor

import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf 

import sklearn.metrics as metrics


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

tf.keras.config.disable_interactive_logging()


target_func = "FUTURE_PERCENT_PROFIT(TQQQ,2)"

df_target = Processor().process(target_func)

print(df_target.head())

best_model = None
best_accuracy = 0
best_functions = None

#in range 1 to 10

FUNCTION_COUNT = 30

for i in range(0,10000):

    #generate the functions
    functions = generate_functions(FUNCTION_COUNT)

    try:
        df_X = Processor().process(functions, autoscale=True)

        #Merge the dataframes
        df = pd.merge(df_X, df_target, on='index_date', how='inner')

        df.dropna(inplace=True)

        #if we have less than 1000 rows, skip
        if df.shape[0] < 200:
            continue

        labels=['Worst','Bad','Neutral','Good','Best']

        df['ProfitCategory'] = pd.qcut(df[target_func], 5, labels=labels)

        #Convert the ProfitCategory to a one-hot encoding
        df = pd.get_dummies(df, columns=['ProfitCategory'])


        df_X = df.drop([target_func,'ProfitCategory_Worst','ProfitCategory_Bad','ProfitCategory_Neutral','ProfitCategory_Good','ProfitCategory_Best'], axis=1)
        df_Y = df[['ProfitCategory_Worst','ProfitCategory_Bad','ProfitCategory_Neutral','ProfitCategory_Good','ProfitCategory_Best']]

        #PCA to reduce the number of features to 10
        pca = PCA(n_components=10)
        df_X = pd.DataFrame(pca.fit_transform(df_X))

        predictions = []
        test_values = []

        model = keras.models.Sequential([
            keras.layers.Input(shape=(10,10,1)),
            keras.layers.Conv2D(50, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(100, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation='sigmoid'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(5, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        WINDOW_SIZE = 250
        for i in range(0, len(df_X) - WINDOW_SIZE - 20):
        # for i in range(0, 2):

            df_X_train = []
            df_Y_train = []

            for j in range(0, WINDOW_SIZE):
                df_X_train.append(df_X.iloc[i+j:i+10+j])
                df_Y_train.append(df_Y.iloc[i+10+j])
                
            df_X_train = np.array(df_X_train)
            df_Y_train = np.array(df_Y_train)

            df_X_test = np.array([df_X.iloc[i+WINDOW_SIZE+1:i+WINDOW_SIZE+11]])
            df_Y_test = np.array([df_Y.iloc[i+WINDOW_SIZE+11]])
            df_Y_test = labels[np.argmax(df_Y_test, axis=1)[0]]

            test_values.append(df_Y_test)

            model.fit(df_X_train, df_Y_train, epochs=10, batch_size=10, verbose=0)

            y_pred = model.predict(df_X_test)

            if i % 100 == 0:
                print(f"{i} of {len(df_X)}")
                print("Predicted: ", labels[np.argmax(y_pred, axis=1)[0]], " Actual: ", df_Y_test)


            #convert the prediction to categorical
            y_pred = labels[np.argmax(y_pred, axis=1)[0]]
            predictions.append(y_pred)          

        test_values = np.array(test_values)
        predictions = np.array(predictions)

        print("*"*80)
        print("Functions")
        print(functions)
        print(metrics.accuracy_score(test_values, predictions))
        print(metrics.confusion_matrix(test_values, predictions))
        print(metrics.classification_report(test_values, predictions))

        if metrics.accuracy_score(test_values, predictions) > best_accuracy:
            best_accuracy = metrics.accuracy_score(test_values, predictions)
            best_model = model
            best_functions = functions

        print("Best Accuracy: ", best_accuracy)
        print("Best Functions: ", best_functions)

    except Exception as e:
        #rethrow the exception
        # raise e
        continue




