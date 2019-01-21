import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
ratios = ["LTC-USD", "ETH-USD", "BTC-USD", "BCH-USD"]

#empty DF
main_df = pd.DataFrame()

for ratio in ratios:
    if len(main_df) == 0:
        temp = pd.read_csv(f"crypto_data/{ratio}.csv", names = ["time", "low", "high", "open", f"{ratio}_close", f"{ratio}_volume"])
        main_df = main_df.append(temp)
        main_df = main_df[["time", f"{ratio}_close", f"{ratio}_volume"]]
    else:
        temp = pd.read_csv(f"crypto_data/{ratio}.csv", names = ["time", "low", "high", "open", f"{ratio}_close", f"{ratio}_volume"])
        temp = temp[["time", f"{ratio}_close", f"{ratio}_volume"]]
        main_df =  pd.merge(main_df, temp, on="time", how="left").fillna(method="ffill")

main_df.set_index("time", inplace=True)

SEQ_LEN = 60 # use 60 mins
FUTURE_PERIOD_PREDICT = 3 # to predict 3 mins
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

    
def preprocess_df(df):
    # useless for us because have "target"
    df = df.drop('future', 1)
    for col in df.columns:
        if col!="target":
            # normalize into percentage, prices are different, only movements are normalized
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            # scale from 0 to 1
            df[col] = preprocessing.scale(df[col].values)
    
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        # transforms row into array
        # think of it this way, after 60 sequences, what is the future in 3 time periods?
        prev_days.append([n for n in i[:-1]])
        print("prev days")
        print(prev_days)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            print("sequetial data")
            print(sequential_data)
            return(sequential_data)
            break
    random.shuffle(sequential_data)
    
main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

# just to be sure our df is sorted
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

#split data
validation_main_df = main_df[(main_df.index) >= last_5pct]
main_df = main_df[(main_df.index) < last_5pct]

print(f"training now has {len(main_df)} rows and validation has {len(validation_main_df)}")

preprocess_df(main_df)
#train_x, train_y = preprocess_df(main_df)
#validation_x, validation_y = preprocess_df(main_df)