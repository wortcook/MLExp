# import torch
import pandas as pd
import numpy as np
from pynoahfunc.processor import Processor
import keras

#set torch device to cpu
class DataSplitter:
    def __init__(self, df : pd.DataFrame, target_col : str, init_size : int, window_size : int):
        self._input_df = df.loc[:, df.columns != target_col]
        self._target_df = df[target_col]
        self._init_size = init_size
        self._window_size = window_size

    def __iter__(self):
        self._curr_idx = self._init_size
        return self
    

    def __next__(self):
        test_target = self._target_df[self._curr_idx+self._window_size]

        #if test_target is empty or Nan then stop iteration
        if pd.isnull(test_target):
            raise StopIteration

        train_input = []
        train_target = []

        for i in range(0, self._curr_idx):
            train_input.append(self._input_df[i:i+self._window_size].values)
            train_target.append(self._target_df[i+self._window_size])

        test_input = self._input_df[self._curr_idx:self._curr_idx+self._window_size].values

        self._curr_idx += 1

        return train_input, [test_input], train_target, [test_target]
    
FUNCTIONS = [
    "QUANTILE_RATIO(C(DGS5),89,0.10,0.90)",
    "RAVG_MEDIAN(L(EURSEKX),8)",
    "RAVG_MEDIAN(C(THREEFYTP3),55)",
    "RAVGS_RATIO(C(DFII30),3,13)",
    "RSEMS_RATIO(C(DEXHKUS),8,8)",
    "RAVG_MEDIAN(V(IAU),34)",
    "RATIO_TO_RAVG(O(IDXT100),34)",
    "RAVGS_RATIO(H(CPER),34,21)",
    "RATIO_TO_RAVG(C(BAMLC0A0CM),21)",
    "RATIO_TO_RAVG(L(CPER),21)",
    "RENTROPY(V(EURGBPX),21)",
    "RATIO_TO_RLINEAR(O(IAU),5)",
    "RKURT(V(IDXVIX),233)",
    "QUANTILE_RATIO(O(PHPX),34,0.25,0.75)",
    "RSI(O(IDXAX),34)",
    "QUANTILE_RATIO(O(IAU),55,0.25,0.75)",
    "RATIO_TO_RAVG(L(EURJPYX),55)",
    "RSEMS_RATIO(H(IDXRUT),144,55)",
    "RENTROPY(V(EURCADX),144)",
    "RAVG_MEDIAN(O(GBPUSDX),5)",
    "RAVGS_RATIO(H(SLV),89,89)",
    "STOCH_OSC_RAVG(AGGDIDL,233,89)",
    "RATIO_TO_RAVG(C(DFF),5)",
    "RSKEW(C(GVZCLS),8)",
    "RAVGS_RATIO(L(CNYX),3,13)",
    "RSI(H(CPER),21)",
    "RENTROPY(V(IDXFCHI),89)",
    "STOCH_OSC_RAVG(TQQQ,21,89)",
    "RENTROPY(L(SPXL),233)",
    "RAVGS_RATIO(C(THREEFYTP7),8,89)",
    "QUANTILE_RATIO(L(IDXFCHI),8,0.25,0.75)",
    "RENTROPY(H(NZDUSDX),5)",
    "RAVG_MEDIAN(O(AGGDIDG),3)",
    "RSI(L(SGDX),233)",
    "QUANTILE_RATIO(V(NZDUSDX),8,0.10,0.90)",
    "RAVGS_RATIO(C(DEXTAUS),13,3)",
    "STOCH_OSC_RAVG(SLV,5,21)",
    "QUANTILE_RATIO(V(IDXAORD),8,0.25,0.75)",
    "RATIO_TO_RAVG(C(IDXFTSE),8)",
    "RAVG_MEDIAN(C(BAMLHE00EHYITRIV),34)",
    "QUANTILE_RATIO(H(EURHUFX),13,0.10,0.90)",
    "RSKEW(C(IUDSOIA),5)",
    "MACDFAST(O(NZDUSDX))",
    "QUANTILE_RATIO(C(AGGDG),144,0.25,0.75)",
    "STOCH_OSC_RAVG(SLV,55,8)",
    "MACDFAST(L(EURJPYX))",
    "W_VOL_AVG(IDXNYA,13,5)",
    "RATIO_TO_RLINEAR(H(IDXBFX),5)",
    "W_VOL_AVG(AGGDIDL,13,144)",
    "RSKEW(C(THREEFF10),3)",
]

TARGET_FUNCTION = "FUTURE_PERCENT_PROFIT(TQQQ,10)"

df = Processor().process("|".join(FUNCTIONS), autoscale = True, scale_type = "robust")
dftarg = Processor().process(TARGET_FUNCTION, autoscale = False)

df = pd.merge(df, dftarg, left_index=True, right_index=True)

#change FUTURE_PERCENT_PROFIT(TQQQ,2) to binary value
df[TARGET_FUNCTION] = df[TARGET_FUNCTION].apply(lambda x: 1 if x > 0 else 0)

splitter = DataSplitter(df, TARGET_FUNCTION, 1000, 50)

siter = iter(splitter)

model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(len(FUNCTIONS), 50,1)),
        keras.layers.Conv2D(500, (50,3), activation="tanh", padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(2000),
        keras.layers.Dense(500, activation="sigmoid"),
        keras.layers.Dense(2, activation="softmax")
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

actuals = []
predictions = []

for train_input, test_input, train_target, test_target in siter:

    train_input = np.array(train_input)
    train_target = np.array(train_target)

    #add column to train_target that is the opposite of the target
    train_target = np.array([[1,0] if x == 0 else [0,1] for x in train_target])

    print(train_input.shape)

    model.fit(train_input, train_target, epochs = 50, batch_size =  int(len(train_input)/10))

    test_input = np.array(test_input)
    test_target = np.array(test_target)

    test_target = np.array([[1,0] if x == 0 else [0,1] for x in test_target])

    test_loss, test_acc = model.evaluate(test_input, test_target)

    actuals.append(test_target[0])
    predictions.append(model.predict(test_input)[0])

    print(f"Expected {test_target[0]}, got {model.predict(test_input)[0]}")

#print confusion matrix
actuals = np.array(actuals)
predictions = np.array(predictions)

confusion_matrix = np.zeros((2,2))

for actual, prediction in zip(actuals, predictions):
    confusion_matrix[actual][prediction] += 1

print(confusion_matrix)





