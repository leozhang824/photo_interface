import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1 - Day
# 2 - Night
# 3 - Noise
# 4 - Not Sure
# 5 - Lack of Data
# 6 - DAQ Unplugged

# classify: use predict but move the voltage into features and make y an np array of types