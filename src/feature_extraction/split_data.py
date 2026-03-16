import os
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split


load_dotenv()
v6_data_dir = os.getenv("V6_FRAMED_DATA")

X = np.load(v6_data_dir)
X_train, X_temp = train_test_split(X, train_size=0.8, random_state=42, shuffle=True)
X_val, X_test = train_test_split(X_temp, train_size=0.5, random_state=42, shuffle=True)


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

np.save("v6_training", X_train)
np.save("v6_validation", X_val)
np.save("v6_testing", X_test)
