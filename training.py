import pandas as pd
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from typing import Protocol, TypedDict
from typing import Any
from provider_types import YAMLconfig
import sys
import torch
import matplotlib.pyplot as plt
import xgboost
from bayes_opt import BayesianOptimization
from yaml import safe_load
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import logging
from logging import Logger
from pathlib import Path
from providers import ArgParseCLI, GraphGenerator, PltPlotter

class TrainingResults(TypedDict):
    r2_score : float
    mape : float
    mae : float

class ModelTrainer(Protocol):
    def train(self)->None:...
    
    def return_training_results(self)->TrainingResults:...

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
    def __init__(self,in_features:int):
        super().__init__()
        assert isinstance(in_features,int), "Argument in_features not of type <int>"
        
        self.fc1 = nn.Linear(in_features=in_features,out_features=in_features * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=in_features * 2,out_features=in_features * 4)
        self.fc3 = nn.Linear(in_features=in_features * 4,out_features=in_features * 2)
        self.fc4 = nn.Linear(in_features=in_features * 2,out_features=in_features)
        self.fc5 = nn.Linear(in_features=in_features,out_features=in_features // 2)
        self.fc6 = nn.Linear(in_features=in_features // 2,out_features=1)
        
        
    def forward(self,input:torch.Tensor):
        output_1 = self.fc1(input)
        output_1 = self.relu(output_1)
        
        output_2 = self.fc2(output_1)
        output_2 = self.relu(output_2)
        
        output_3 = self.fc3(output_2)
        output_3 = self.relu(output_3)
        output_3 = output_3 + output_1
        
        output_4 = self.fc4(output_3)
        output_4 = self.relu(output_4)
        output_4 = output_4 + input
        
        output_5 = self.fc5(output_4)
        output_5 = self.relu(output_5)
        
        output_6 = self.fc6(output_5)
        return output_6

def train_dl_model(train_loader:DataLoader,test_loader:DataLoader,hyper_parameters:HyperParameters,feature_dim:int,logger:Logger, grapher:GraphGenerator)->tuple:
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
    5. logger : ``Logger``
        - Used to log data
    
    ### Returns
    The last r2 score and MAPE score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(feature_dim).to(device=device)
    validation_mape = 0
    validation_r2_score = 0
    training_mape_scores = []
    validation_mape_scores = []
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=hyper_parameters.get_learning_rate(),
        weight_decay=hyper_parameters.get_weight_decay()
    )
    loss_function = nn.HuberLoss().to(device)
            
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
        
        training_mape = mean_absolute_percentage_error(training_targets_ndarray,training_predictions_ndarray)
        training_r2 = r2_score(training_targets_ndarray,training_predictions_ndarray)
        training_mape_scores.append(training_mape)
        
        logger.info(f"Epoch {i+1}: Training R2: {training_r2} Training MAPE: {training_mape}")
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
        validation_mape_scores.append(validation_mape)
        validation_r2_score = r2_score(validation_targets_ndarrray,validation_predictions_ndarrray)
        
        logger.info(f"Epoch {i+1}: Validation R2: {validation_r2_score} Validation MAPE: {validation_mape}\n\n")

    grapher.create_single_metric_comparison_graph(
        save_path = Path('./data/training_validation_mape_graph.png'),
        shared_x_values = [i+1 for i in range(len(training_mape_scores))],
        metric_name = 'Mean Average Percent Error',
        series_values = [
            training_mape_scores,
            validation_mape_scores
        ],
        series_labels = [
            'Training',
            'Validation'
        ]
    )
    
    return validation_r2_score,validation_mape

def preprocess_data(data_df : pd.DataFrame, scaler: StandardScaler,target_col:str,feature_cols:list[str],training_set=False,)->tuple[np.ndarray,np.ndarray]:
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
    
def train_test_split_df(file_path:Path,train_split:float)->tuple[pd.DataFrame,pd.DataFrame]:
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


def return_yaml_config(path:Path)->YAMLconfig:
    with open(path) as f:
        config = YAMLconfig.from_yaml(f)
        if isinstance(config,list):
            raise ValueError("YAML Config returned as a list of objects")
        return config

def return_logger(logging_folder:Path,log_file_name:str)->Logger:
    log_path = logging_folder / f'{log_file_name}.log'
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path,filemode='w',datefmt='%m-%d-%Y %H-%M',level=logging.INFO)
    
    return logger

if __name__ == "__main__":
    cli = ArgParseCLI()
    fields = cli.get_cli_arguments()
    yaml_config : YAMLconfig = return_yaml_config(fields.yaml_file_location)
    log_folder = fields.log_folder
    experiment_name = fields.experiment_name
    run_name = fields.run_name
    
    experiment_log_folder = log_folder / experiment_name
    if not experiment_log_folder.exists():
        experiment_log_folder.mkdir()
        
    logger = return_logger(experiment_log_folder,run_name)
    
    training_split = 0.8
    
    target_column_name = yaml_config.training_details.target_column
    data_file = Path(yaml_config.data_file_name)
    feature_columns = yaml_config.training_details.feature_columns
    tracking_uri = yaml_config.ml_flow_endpoint
        
    
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
    batch_size = yaml_config.hyper_parameters.batch_size
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    
    bounds = {
        'n_estimators' : (100,500),
        'max_depth' : (1,1000),
        'min_samples_split' : (2,100),
        'min_samples_leaf' : (0.01,0.99),
        'min_weight_fraction_leaf' : (0,0.5)
    }
    
    def optimize_r2(n_estimators,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            criterion='squared_error'
        )
        model.fit(x_train,y_train)
        r2 = r2_score(y_test,model.predict(x_test))
        return r2
    
    optim = BayesianOptimization(
        f=optimize_r2,
        pbounds=bounds,
        random_state=1
    )
    
    # optim.maximize(
    #     init_points=100,
    #     n_iter=100,
        
    # )
    
    
    
    with mlflow.start_run(run_name=run_name) as run:
        
        hyper_parameters = HyperParameters(
            learning_rate= yaml_config.hyper_parameters.learning_rate,
            weight_decay= yaml_config.hyper_parameters.weight_decay,
            epochs= yaml_config.hyper_parameters.epochs
        )
        
        mlflow.log_params(hyper_parameters.get_params())
        mlflow.log_param('batch_size',batch_size)
        
        train_dataloader : DataLoader = return_dataloader(x_train,y_train,batch_size=batch_size)
        test_dataloader : DataLoader = return_dataloader(x_test,y_test,batch_size=batch_size)
        
        plt_plotter = PltPlotter()
        r2_metric, mape_metric = train_dl_model(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            hyper_parameters=hyper_parameters,
            feature_dim=x_train.shape[-1],
            logger=logger,
            grapher = plt_plotter
        )
        
        mlflow.log_metric('r2',r2_metric)
        mlflow.log_metric('mape',mape_metric)
        for graph_path in plt_plotter.get_create_graphs():
            mlflow.log_artifact(str(graph_path))