# data_loader.py
from ucimlrepo import fetch_ucirepo

def load_data():
    # fetch dataset
    combined_cycle_power_plant = fetch_ucirepo(id=294)
    
    # data (as pandas dataframes)
    X = combined_cycle_power_plant.data.features
    y = combined_cycle_power_plant.data.targets
    
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print(X.head())
    print(y.head())