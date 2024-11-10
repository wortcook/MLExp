import random

from pynoahdb.symbolseries import SymbolSeries
from pynoahdb.algo_feature_data import AlgoFeatureData
from pynoahfunc.processor import Processor
from pywortutil.algos import Genetic, GeneticCallback


import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf 

#import support vector machine models
from sklearn.svm import LinearSVC as SVC

import sklearn.metrics as metrics


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA


#suppress warnings
import warnings
warnings.filterwarnings("ignore")


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

# SYMBOL_TYPES = ["EQUITY","INDEX","ETF","CURRENCY","FRED","AGGREGATE"]
SYMBOL_TYPES = ["INDEX","ETF","CURRENCY","FRED","AGGREGATE"]



EQUITY_SYMBOLS = SymbolSeries().symbol_list("EQUITY")
INDEX_SYMBOLS = SymbolSeries().symbol_list("INDEX")
ETF_SYMBOLS = SymbolSeries().symbol_list("ETF")
CURRENCY_SYMBOLS = SymbolSeries().symbol_list("CURRENCY")
FRED_SYMBOLS = SymbolSeries().symbol_list("FRED")
AGGREGATE_SYMBOLS = SymbolSeries().symbol_list("AGGREGATE")

def generate_functions(f_count):
    #Get the symbols
    ret_list = []

    selected_functions = 0

    while selected_functions < f_count:
        #Get the function
        function = random.choice(FUNCTIONS)

        sym_type = random.choice(SYMBOL_TYPES)

        #Get the symbol
        if sym_type == "EQUITY":
            symbol = random.choice(EQUITY_SYMBOLS)
        elif sym_type == "INDEX":
            symbol = random.choice(INDEX_SYMBOLS)
        elif sym_type == "ETF":
            symbol = random.choice(ETF_SYMBOLS)
        elif sym_type == "CURRENCY":
            symbol = random.choice(CURRENCY_SYMBOLS)
        elif sym_type == "FRED":
            symbol = random.choice(FRED_SYMBOLS)
        elif sym_type == "AGGREGATE":
            symbol = random.choice(AGGREGATE_SYMBOLS)

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

        try:
            print(function[0].format(*params))
            ret_list.append(Processor().process(function[0].format(*params), autoscale = True, scale_type = "robust"))
            selected_functions += 1
        except:
            continue

    return ret_list

def evaluate_functions(df_X_arr, show_metrics = False):
    try:
        df = df_X_arr[0]
        for i in range(1,len(df_X_arr)):
            df = pd.merge(df, df_X_arr[i], on='index_date', how='inner')

        #Merge the dataframes
        df = pd.merge(df, DF_TARGET, on='index_date', how='inner')

        df.dropna(inplace=True)

        #if we have less than 1000 rows, skip
        if df.shape[0] < 500:
            return -1.0
        
        Y = df[target_func]
        X = df.drop(columns=[target_func])

        Y_pos = Y[Y == 1].count()
        Y_neg = Y[Y == 0].count()

        #Compute regression weights
        weights = {0:Y_pos/(Y_pos+Y_neg), 1:Y_neg/(Y_pos+Y_neg)}

        #Split the data into the last 10% for testing
        split_index = int(len(df) * 0.8)

        X_train = X[:split_index]
        Y_train = Y[:split_index]

        X_test = X[split_index:]
        Y_test = Y[split_index:]

        #Create the model
        model = SVC(class_weight=weights)

        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        score = metrics.accuracy_score(Y_test, Y_pred)

        if score > 0.65:
            for dc in df_X_arr:
                AlgoFeatureData().add_score('LinearSVC', dc.columns[0], score)

        if show_metrics:
            print(metrics.classification_report(Y_test, Y_pred))
            print(metrics.confusion_matrix(Y_test, Y_pred))
            print(f"Accuracy: {score}")
        
        return score
        
    except Exception as e:
        print(e)
        raise e
        # return -1.0
    
def gene_hash(gene):
    gen_hash = ""
    for g in gene:
        gen_hash += g.columns[0]
        
    return gen_hash

target_func = "FUTURE_PERCENT_PROFIT(TQQQ,2)"
DF_TARGET = Processor().process(target_func)

DF_TARGET[target_func] = DF_TARGET[target_func].apply(lambda x: 1 if x > 0 else 0)

class GCallback(GeneticCallback):
    def __init__(self):
        self._gen_count = 0

    def run_generation(self, population):
        self._gen_count += 1
        print(f"Generation: {self._gen_count}")
        print(f"Population size: {len(population)}")
        print("*"*80)

        #Get best results
        evaluate_functions(population[0][0], show_metrics = True)
        print("*"*80)

        


algo = Genetic(generate_functions, evaluate_functions, gene_hash, cull_rate = 0.5, dominant_rate = 0.5, mutation_rate = 0.05, genetic_size = 10, callback = GCallback())

population = algo.run(1000, initial_population = 500, stable_population_size = 100)

#sort the population by the fitness
population.sort(key=lambda x: x[1], reverse=True)

#print the columns of the best 10 items and the score
for i in range(10):
    print(population[i][1])
    functions = population[i][0]
    flist = []
    for f in functions:
        flist.append(f.columns[0])
    print("|".join(flist))

#show the metrics
evaluate_functions(population[0][0], show_metrics = True)