import pandas as pd
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import sys
import torch
import xgboost
from sklearn.metrics import mean_absolute_percentage_error, r2_score
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

    def get_params(self)->dict:
        return {
            'learning_rate' : self.learning_rate,
            'weight_decay' : self.weight_decay,
            'epochs' : self.epochs
        }

class NeuralNetwork(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features,out_features=1)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=256,out_features=128)
        # self.fc3 = nn.Linear(in_features=128,out_features=64)
        # self.fc4 = nn.Linear(in_features=64,out_features=32)
        # self.fc5 = nn.Linear(in_features=32,out_features=16)
        # self.fc6 = nn.Linear(in_features=16,out_features=1)
        
    def forward(self,x:torch.Tensor):
        x = self.fc1(x)
        # x = self.relu(x)
        
        # x = self.fc2(x)
        # x = self.relu(x)
        
        # x = self.fc3(x)
        # x = self.relu(x)
        
        # x = self.fc4(x)
        # x = self.relu(x)
        
        # x = self.fc5(x)
        # x = self.relu(x)
        
        # x = self.fc6(x)
        return x

def train_dl_model(train_loader:DataLoader,test_loader:DataLoader,hyper_parameters:HyperParameters,feature_dim:int)->tuple:
    """
    Train the DL model using the provided data and hyper parameters.
    ### Parameters
    1. train_loader : ``DataLoader``
        - Contains the training samples
    2. test_loader : ``DataLoader``
        - Contains the test samples.
    3. hyper_parameters : ``HyperParameters``
        - Hyper parameters used for training
    4. feature_dim : ``int``
        - Specifies the dimension of the input features
    
    ### Returns
    The last r2 score and MAPE score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(feature_dim).to(device=device)
    validation_mape = 0
    validation_r2_score = 0
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=hyper_parameters.get_learning_rate(),
        weight_decay=hyper_parameters.get_weight_decay()
    )
    loss_function = nn.HuberLoss().to(device)
        
    logging.basicConfig(level=logging.INFO)
    
    for i in range(hyper_parameters.get_epochs()):
        training_predictions = []
        training_targets = []
        
        validation_predictions = []
        validation_targets = []
        
        model.train()
        for features, targets in train_loader:
            optim.zero_grad()
            features = features.to(device)
            targets : torch.Tensor = targets.to(device)
            prediction : torch.Tensor = model(features)
            training_predictions.append(prediction.detach().cpu().numpy())
            training_targets.append(targets.detach().cpu().numpy())
            loss = loss_function(prediction,targets)
            loss.backward()
            optim.step()
        
        training_predictions_ndarray = np.concatenate(training_predictions,axis=0)
        training_targets_ndarray = np.concatenate(training_targets,axis=0)
        
        training_predictions_ndarray = training_predictions_ndarray.reshape(training_predictions_ndarray.shape[0])
        training_targets_ndarray = training_targets_ndarray.reshape(training_targets_ndarray.shape[0])
        
        training_mae = mean_absolute_percentage_error(training_targets_ndarray,training_predictions_ndarray)
        
        logging.info(msg=f"Training MAPE: {training_mae}, epoch: {i+1}")
        model.eval()
        for features, targets in test_loader:
            features = features.to(device)
            targets : torch.Tensor = targets.to(device)
            prediction : torch.Tensor = model(features)
            validation_predictions.append(prediction.detach().cpu().numpy())
            validation_targets.append(targets.detach().cpu().numpy())
        
        validation_predictions_ndarrray = np.concatenate(validation_predictions,axis=0)
        validation_targets_ndarrray = np.concatenate(validation_targets,axis=0)    
        
        validation_predictions_ndarrray = validation_predictions_ndarrray.reshape(validation_predictions_ndarrray.shape[0])
        validation_targets_ndarrray = validation_targets_ndarrray.reshape(validation_targets_ndarrray.shape[0])
        
        validation_mape = mean_absolute_percentage_error(validation_targets_ndarrray,validation_predictions_ndarrray)
        validation_r2_score = r2_score(validation_targets_ndarrray,validation_predictions_ndarrray)
        
        print(f'Epoch {i+1}',f'R2: {validation_r2_score}',f'MAPE: {validation_mape}')

    return validation_r2_score,validation_mape

def preprocess_data(data_df : pd.DataFrame, scaler: StandardScaler,training_set:False,target_col:str,feature_cols:list[str])->tuple[np.ndarray,np.ndarray]:
    """
    Given the dataframe, return the target values, and features in a tuple after the application of preprocessing. 
    
    ### Parameters
    1. data_df : ``pd.DataFrame``
        - DataFrame contained both the features and target value.
    2. scaler: ``StandardScaler``
        - Scaler used to fit the training data, and transform the the training and test data.
    3. training_set: ``bool``
        - If ``True``, fits the scaler on the data and transforms it. Otherwise, simply transforms the data.
    4. target_col : ``str``
        - Name of the column containing the target values.
    5. features_cols : ``list[str]``
        - List of column names containing the features
    
    ### Returns
    Tuple in the form ``(features_ndarray, target_ndarray)``.
    """
    target_ndarray = data_df[target_col].to_numpy().reshape(-1,1)
    features_df = data_df[feature_cols]
    
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
    data_file = "./data/Node Information Rows Removed.xlsx"
    training_split = 0.8
    
    if len(sys.argv) == 1:
        raise Exception("Please also pass in the name of the run as a command line argument")
    
    target_column_name = "target_traffic_count"
    feature_columns = [
        'Source Local','source_traffic_count','Summed Centrality',
        'Access Point Local Centrality Scores',
       'Access Point Collector Centrality Scores',
       'Access Point Minor Arterial Centrality Scores',
       'Access Point Other Centrality Scores',
       'Access Point Major Arterial Centrality Scores',
       'Source Collector', 'Source Minor Arterial', 'Source Other',
       'Source Major Arterial', 'Source speed', 'Target Local',
       'Target Collector', 'Target Minor Arterial', 'Target Other',
       'Target Major Arterial', 'Target speed','Total Path Distance',
       'Access Points Local', 'Access Points Collector',
       'Access Points Minor Arterial', 'Access Points Other',
       'Access Points Major Arterial','Path Local', 'Path Collector',
       'Path Minor Arterial', 'Path Other', 'Path Major Arterial'
    ]
        
    
    train_split_df, test_split_df = train_test_split_df(data_file,train_split=training_split)
    
    scaler = StandardScaler()
    
    x_train, y_train = preprocess_data(train_split_df,scaler=scaler,training_set=True,target_col=target_column_name,feature_cols=feature_columns)
    x_test, y_test = preprocess_data(test_split_df,scaler=scaler,training_set=False,target_col=target_column_name,feature_cols=feature_columns)
    
    # regressor = xgboost.XGBRegressor()
    # regressor.fit(x_train,y_train),
    # score = regressor.score(x_test,y_test)
    # mape = mean_absolute_percentage_error(y_test,regressor.predict(x_test))
    # print(mape)
    # print(score)
    # Deep Learning configuration
    batch_size = 16
    
    mlflow.set_experiment("AAWDT Modulation - Deep Learning")
    
    with mlflow.start_run(run_name=sys.argv[-1]):
        hyper_parameters = HyperParameters(
            learning_rate=0.0005,
            weight_decay=0.0,
            epochs=100
        )
        mlflow.log_params(hyper_parameters.get_params())
        mlflow.log_param('batch_size',batch_size)
        
        train_dataloader : DataLoader = return_dataloader(x_train,y_train,batch_size=batch_size)
        test_dataloader : DataLoader = return_dataloader(x_test,y_test,batch_size=batch_size)
        
        r2_metric, mape_metric = train_dl_model(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            hyper_parameters=hyper_parameters,
            feature_dim=x_train.shape[-1]
        )
        
        mlflow.log_metric('r2',r2_metric)
        mlflow.log_metric('mape',mape_metric)