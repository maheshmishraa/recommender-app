import pandas as pd
def load_data(path):
    data = pd.read_csv(path)
    
    #Remove unnecessary columns
    data=data.drop(columns=['Unnamed: 0'],errors='ignore')
    
    #Drop rows with missing values
    data=data.dropna(subset=['title','overview'])
    
    return data

def create_features(data):
    #only overview is strong signal
    data["combined"]=data["overview"]
    
    return data