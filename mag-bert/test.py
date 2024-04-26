import pandas as pd
import os

def _read_csv(input_file):
        """Read csv file. Return pandas dataframe."""
        inputs = pd.read_csv(input_file, sep=',')
        return inputs

inputs = _read_csv(os.path.join("datasets", "MIntRec", "relations", "atomic_test.csv"))
            
for i, line in enumerate(inputs['xAttr']):
    print(line[2:len(line)-2])
    if i == 3:
        break