import pandas as pd
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import torch 
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import logging

class HyperParameters():
    def __init__(self,learning_rate,weight_decay,epochs):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_weight_decay(self):
        return self.weight_decay
    
    def get_epochs(self):
        return self.epochs

class NeuralNetwork(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=18,out_features=180),
            nn.ReLU(),
            nn.Linear(in_features=180,out_features=90),
            nn.ReLU(),
            nn.Linear(in_features=90,out_features=45),
            nn.ReLU(),
            nn.Linear(in_features=45,out_features=15),
            nn.ReLU(),
            nn.Linear(in_features=15,out_features=1)
        )
    
    def forward(self,x:torch.Tensor):
        return self.model(x)

def train_dl_model(train_loader:DataLoader,test_loader:DataLoader,hyper_parameters:HyperParameters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork().to(device=device)
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=hyper_parameters.get_learning_rate(),
        weight_decay=hyper_parameters.get_weight_decay()
    )
    loss_function = nn.MSELoss().to(device)
        
    logging.basicConfig(level=logging.INFO)
    
    for i in range(hyper_parameters.get_epochs()):
        training_predictions = []
        training_targets = []
        
        validation_predictions = []
        validation_targets = []
        
        for targets, features in train_loader:
            features = features.to(device)
            targets : torch.Tensor = targets.to(device)
            prediction : torch.Tensor = model(features)
            training_predictions.append(prediction.detach().cpu().numpy())
            training_targets.append(targets.detach().cpu().numpy())
            loss = loss_function(prediction,targets)
            loss.backward()
            optim.step()
            optim.zero_grad()
        
        training_predictions_ndarray = np.concatenate(training_predictions,axis=0)
        training_targets_ndarray = np.concatenate(training_targets,axis=0)
        
        training_predictions_ndarray = training_predictions_ndarray.reshape(training_predictions_ndarray.shape[0])
        training_targets_ndarray = training_targets_ndarray.reshape(training_targets_ndarray.shape[0])
        
        training_mae = mean_absolute_percentage_error(training_targets_ndarray,training_predictions_ndarray)
        
        logging.info(msg=f"Training MAPE: {training_mae}, epoch: {i+1}")
        for targets, features in test_loader:
            features = features.to(device)
            targets : torch.Tensor = targets.to(device)
            prediction : torch.Tensor = model(features)
            validation_predictions.append(prediction.detach().cpu().numpy())
            validation_targets.append(targets.detach().cpu().numpy())
        
        validation_predictions_ndarrray = np.concatenate(validation_predictions,axis=0)
        validation_targets_ndarrray = np.concatenate(validation_targets,axis=0)    
        
        validation_predictions_ndarrray = validation_predictions_ndarrray.reshape(validation_predictions_ndarrray.shape[0])
        validation_targets_ndarrray = validation_targets_ndarrray.reshape(validation_targets_ndarrray.shape[0])
        
        validation_mae = mean_absolute_percentage_error(validation_targets_ndarrray,validation_predictions_ndarrray)
    
        logging.info(msg=f"Validation MAPE: {validation_mae}, epoch: {i+1}")

def preprocess_data(data_df : pd.DataFrame, scaler: StandardScaler,training_set:False)->tuple[np.ndarray,np.ndarray]:
    """
    Given the dataframe, return the target values, and features in a tuple after the application of preprocessing. 
    
    ### Parameters
    1. data_df : ``pd.DataFrame``
        - DataFrame contained both the features and target value.
    
    ### Returns
    Tuple in the form ``(features_ndarray, target_ndarray)``.
    """
    target_ndarray = data_df['target_traffic_count'].to_numpy().reshape(-1,1)
    features_df = data_df.drop(labels=['target_traffic_count'],axis=1)
    
    if training_set:
        features_ndarray = scaler.fit_transform(features_df)
    else:
        features_ndarray = scaler.transform(features_df)

    return features_ndarray.astype(np.float32), target_ndarray.astype(np.float32),
    
def train_test_split_df(file_path:str,train_split:float)->tuple[pd.DataFrame,pd.DataFrame]:
    """
    Given the file path, create a DataFrame object and split it into two based on the split threshold, and return the
    split DataFrames.
    
    ### Parameters
    1. file_path : ``str``
        - File path specifying the location of the dataset.
    2. train_split : ``float``
        - The percentage, given as a float, to be used for the training set.
        
    ### Returns
    A tuple returning the split data as ``(training_dataset_df, test_dataset_df)``.
    """
    data_df = pd.read_excel(file_path)
    
    last_training_index = int(data_df.shape[0] * train_split)
    
    training_data_df = data_df[:last_training_index]
    
    test_data_df = data_df[last_training_index :]
    
    return training_data_df, test_data_df

def return_dataloader(features:np.ndarray,targets:np.ndarray,batch_size:int)->DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets)
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

if __name__ == "__main__":
    data_file = "./data/Features_functional_road_classes.xlsx"
    training_split = 0.8
    
    train_split_df, test_split_df = train_test_split_df(data_file,train_split=training_split)
    
    scaler = StandardScaler()
    
    x_train, y_train = preprocess_data(train_split_df,scaler=scaler,training_set=True)
    x_test, y_test = preprocess_data(test_split_df,scaler=scaler,training_set=False)
    
    regressor = GradientBoostingRegressor(loss='huber',n_estimators=200,learning_rate=0.02)
    regressor.fit(x_train,y_train),
    score = regressor.score(x_test,y_test)
    print(score)
    # # Deep Learning configuration
    # batch_size = 16
    
    # hyper_parameters = HyperParameters(
    #     learning_rate=0.005,
    #     weight_decay=0.0001,
    #     epochs=100
    # )
    
    # train_dataloader : DataLoader = return_dataloader(x_train,y_train,batch_size=batch_size)
    # test_dataloader : DataLoader = return_dataloader(x_test,y_test,batch_size=batch_size)
    
    # train_dl_model(
    #     train_loader=train_dataloader,
    #     test_loader=test_dataloader,
    #     hyper_parameters=hyper_parameters
    # )