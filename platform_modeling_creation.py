import pandas as pd
import ast

ROAD_CLASS_COLUMN_NAME = "roadclass"
SPEED_COLUMN_NAME = "speed"
SOURCE_MIOVOSION_COL_NAME ="source_miovision_id"
TARGET_MIOVSION_COL_NAME = "target_miovision_id"
NODE_ID_COL_NAME = "NODEID"
EDGE_ID_COL_NAME = "EDGEID"
EDGE_LIST_PER_NODE_COL_NAME = "connected_edges"
NODES_PASSED_COLUMN_NAME = "nodes_passed"

def return_summed_centrality(node_info_df:pd.DataFrame,nodes:list)->float:
    """
    Given nodes in the list, find and sum up the centrality scores from the corresponding rows in the node_info_df.
    
    ### Parameters
    1. node_info_df : ``pandas.DataFrame``
        - The DataFrame object that contains information per node. 
    2. nodes : ``list``
        - List of nodes visited in the path.
        - The first and last node in the list correspond to the origin and destination. 
    
    ### Returns
    The value of the summed centrality scores, stored as a ``float`` object. 
    """
    
    nodes = nodes[1: len(nodes) - 1]
    selected_subset = node_info_df[node_info_df[NODE_ID_COL_NAME].isin(nodes)]
    
    return selected_subset['centrality_original'].sum()

def return_aggregate_path_road_class_counts(edges_info_df:pd.DataFrame,edges:list,prefix:str)->dict:
    """
    Given a list of edges on a path, perform aggregation over the occurences of the road 
    classes across the edges using information provided in the edges information dataframe.
    
    ### Parameters
    1. edges_info_df: ``pd.DataFrame``
        - The DataFrame object containing road class information for each index
    2. edges: ``list``
        - The list of edges provided.
    3. prefix: ``str``
        - The prefix to be added to the name of each unique road class. 
    ### Returns
    A ``dict`` object containing road class-count key-value pairs
    """
    road_class_col_name = ROAD_CLASS_COLUMN_NAME
    unique_road_classes = edges_info_df[road_class_col_name].unique()
    if len(edges) == 0:
        return_dict = {f'{prefix} {road_class}':0 for road_class in unique_road_classes}
        return return_dict
    else:
        filtered_edges_info_df = edges_info_df[edges_info_df[EDGE_ID_COL_NAME].isin(edges)]
        road_class_counts = filtered_edges_info_df[road_class_col_name].value_counts()        
        return {f'{prefix} {road_class}':road_class_counts.get(road_class,0) for road_class in unique_road_classes}

def return_access_points_edges(node_info_df:pd.DataFrame,edges:list,nodes:list)->list:
    """
    Given the lists of edges and nodes, generate a list containing the edges that were adjacent to the path, acting
    as access points that could disrupt or affect the volume along the path. 
    
    ### Parameters
    1. edges : ``list``
        - List of edges visited in the path.
        - Containing n + 1 edges given n visited nodes.
    2. nodes : ``list``
        - List of nodes visited in the path.
        - The first and last node in the list correspond to the origin and destination.
    3. node_info_df : ``pandas.DataFrame``
        - The DataFrame object containing the list of adjacent edges per node.
    
    ### Returns
    A ``list`` object containing the id's of the adjacent edges to the path. 
    """
    node_info_df = node_info_df[[NODE_ID_COL_NAME,"connected_edges"]].copy()
    
    node_index = 1
    last_node_index = len(nodes) - 1
    
    nodes_visited = nodes[1:last_node_index]
    edges_visited = []
    while node_index < last_node_index:
        edges_visited.append({edges[node_index - 1],edges[node_index]})
        node_index += 1
    
    nodes_edges_visited_df = pd.DataFrame(data={NODE_ID_COL_NAME:nodes_visited,"visited_edges":edges_visited})
    final_list_access_points = []
    
    # The only way that the shape would be 0 is if there were only two nodes in the path (the origin and the des),
    # Meaning that no access points exist on the path
    if nodes_edges_visited_df.shape[0] != 0:
        merged_nodes_info_df = nodes_edges_visited_df.merge(right=node_info_df,on=NODE_ID_COL_NAME)
        result = merged_nodes_info_df.apply(lambda x : list(x['visited_edges'].symmetric_difference(x['connected_edges'])),axis=1)
        merged_nodes_info_df['access_points'] = result
        for access_point_list in merged_nodes_info_df['access_points'].tolist():
            final_list_access_points.extend(access_point_list)
    
    return final_list_access_points

def return_centrality_scores_by_road_class(edge_info_df:pd.DataFrame,edges:list)->dict[str,float]:
    """
    Given the information stored in the provided dataframe, aggregate the centrablity scores by each road class
    provided in the list of edges.
    
    ### Parameters
    1. edge_info_df : ``pandas.DataFrame``
        - Dataframe object containing information for the edges.
    2. edges : ``list``
        - List of edges over which to aggregate the centrality scores by road class
        
    ### Returns
    A dict containing the road classes as keys and the summed centrality scores for each on the given path as keys.
    """
    road_class_column = ROAD_CLASS_COLUMN_NAME
    centrality_scores_column = "centrality_original"
    id_column = EDGE_ID_COL_NAME
    unique_road_classes = edge_info_df[road_class_column].unique()
    
    if len(edges) == 0:
        return {f'Access Point {unique_class} Centrality Scores' : 0 for unique_class in unique_road_classes}
    else:
        filtered_df = edge_info_df[edge_info_df[id_column].isin(edges)]
        aggregate_scores = filtered_df.groupby(road_class_column)[centrality_scores_column].sum()
        return {f'Access Point {unique_class} Centrality Scores' : aggregate_scores.get(unique_class,0) for unique_class in unique_road_classes}
def return_path_distance(edge_info_df:pd.DataFrame,edges:list)->float:
    """
    Cross reference the length information in the DataFrame with the edges in the list and summate the total
    distance across the edges provided in the list.
    
    ### Parameters:
    1. edge_info_df : ``pandas.DataFrame``
        - The DataFrame object containing the information for all edges
    2. edges : ``list``
        - The list containing edges
        
    ### Returns
    The total summed up distance as a float.
    """
    
    if len(edges) == 0:
        return 0
    else:
        edges_information = edge_info_df['length'][edge_info_df[EDGE_ID_COL_NAME].isin(edges)]
        return edges_information.sum()
    
def return_edge_centrality_summation(edges:list[int],edge_information_df:pd.DataFrame)->float:
    """
    Given the centrality score listed for all edges in the information data, return a summation for the edges
    in the list provided.
    
    ### Parameters:
    1. edges: ``list[int]``
        - List of edges used for filtering the dataframe.
    2. edge_information_df : ``pandas.DataFrame``
        - The dataframe object containing information per edge
    
    ### Returns
    The total summed edge_centrality scores for the edges provided.
    """
    if len(edges) == 0:
        return 0
    else:
        filtered_subset_indices = edge_information_df[EDGE_ID_COL_NAME].isin(edges)
        return edge_information_df['centrality_original'][filtered_subset_indices].sum()
    
def return_edge_features(edge_id:int,edge_information_df:pd.DataFrame)->pd.Series:
    """
    Given the EDGE ID, return a Series object using a dictionary mapping containing two key-value pairs. The first pair contains the road class column as a key and the name of the column as value.
    The second pair contains the name of the speed column and the corresponding speed value.
    
    ### Parameters
    1. edge_id : ``int``
        - The ID of the edge for which information will be extracted.
    2. edge_information_df : ``pd.DataFrame``
        - The DataFrame containing the features per edge.
    
    ### Returns
    The ``Series`` object containing that information described above.
    """
    return_dict = {}
    
    try:
        return_dict[ROAD_CLASS_COLUMN_NAME] = edge_information_df[ROAD_CLASS_COLUMN_NAME][edge_information_df[EDGE_ID_COL_NAME] == edge_id].to_list()[0]
        return_dict[SPEED_COLUMN_NAME] = edge_information_df[SPEED_COLUMN_NAME][edge_information_df[EDGE_ID_COL_NAME] == edge_id].to_list()[0]
    except Exception as e:
        print('Caused by edge_id {id}'.format(id=edge_id))
        raise e
    
    return pd.Series(return_dict)


def aggregate_edge_features(edge_ids:list[int],edge_information_df:pd.DataFrame,prefix:str)->dict[str,float]:
    """
    Given the list of edge_ids, aggregate the edge features for each using the edge_information_df and appending the prefix onto each key name.
    
    ### Parameters
    1. edge_ds : ``list[int]``
        - List of edge ids to be aggregated over.
    2. edge_information_df : ``DataFrame``
        - contains edge information
    3. prefix : ``str``
        - Prepended to each column name
        
    ### Returns
    A ``dict`` object with aggregated features
    """
    unique_road_classes = edge_information_df[ROAD_CLASS_COLUMN_NAME].unique()
    return_dict =  {f'{prefix} {unique_class}' : 0 for unique_class in unique_road_classes}
    return_dict[f'{prefix} {SPEED_COLUMN_NAME}'] = 0

    edge_features_list = []
    for edge_id in edge_ids:
        edge_features_series = return_edge_features(edge_id=edge_id,edge_information_df=edge_information_df)
        edge_features_list.append(edge_features_series)
    
    edge_features_df = pd.DataFrame(edge_features_list)
    road_classes_count = edge_features_df.groupby(ROAD_CLASS_COLUMN_NAME)[SPEED_COLUMN_NAME].count()
    average_speed = edge_features_df[SPEED_COLUMN_NAME].mean()
    
    return_dict[f'{prefix} {unique_class}'] = average_speed
    for unique_class in unique_road_classes:
        return_dict[f'{prefix} {unique_class}'] = road_classes_count.get(unique_class,0)
    
    return return_dict
def return_node_features(node_id:int,node_information_df:pd.DataFrame,edge_information_df:pd.DataFrame,prefix:str)->dict[str,float]:
    """
    Given the node ID, aggregate the road class and speed values across all adjacent edges of the corresponding node and 
    return a dict mapping the value of each feature with a prepended prefix to the corresponding features.
    
    ### Parameters
    1. node_int : ``int``
        - ID used to get the list of edges from the node_information_df
    2. node_information_df : ``DataFrame``
        - Contains the list of edges connected to each node.
    3. edge_information_df : ``DataFrame``
    """
    node_information_filtered_df = node_information_df[node_information_df[NODE_ID_COL_NAME] == node_id]
    
    assert node_information_filtered_df.shape[0] == 1, "There should exist only one NODE for the given NODEID. More than one was found."
    
    connected_edges = node_information_filtered_df[EDGE_LIST_PER_NODE_COL_NAME].tolist()[0]
    return aggregate_edge_features(edge_ids=connected_edges,edge_information_df=edge_information_df,prefix=prefix)
    

def create_features() -> pd.DataFrame:
    """
    Utilizes the following files for creation of the features using the information on the edges and the nodes
    on each path:
    1. ``./data/edge_info.csv``
    2. ``./data/node_info.csv``
    3. ``./data/training_paths.csv``
    ### Returns
    A ``pandas.DataFrame`` object containing the necessary features.
    """
    # Upload all dataframes
    training_paths_df = pd.read_csv('./data/training_paths.csv')
    node_info_df = pd.read_csv('./data/node_info.csv')
    edge_info_df = pd.read_csv('./data/edge_info.csv')
    
    # Turn string into list for appropriate columns
    training_paths_df['edges_passed'] = training_paths_df['edges_passed'].apply(ast.literal_eval)
    training_paths_df['nodes_passed'] = training_paths_df['nodes_passed'].apply(ast.literal_eval)
    
    node_info_df['connected_edges'] = node_info_df['connected_edges'].apply(ast.literal_eval)
    
    # Add list of access point edges for each path
    training_paths_df['access_points_edges'] = training_paths_df.apply(lambda x: return_access_points_edges(node_info_df,x['edges_passed'],x['nodes_passed']),axis=1)

    # Add the summation of the centrality scores
    training_paths_df['Summed Centrality'] = training_paths_df.apply(lambda x: return_summed_centrality(node_info_df,x['nodes_passed']),axis=1)
    
    # Add the total path distance
    training_paths_df['Total Path Distance'] = training_paths_df.apply(lambda x: return_path_distance(edge_info_df,x['edges_passed']),axis=1)
    
    # Add the summed centrality measures
    # training_paths_df['Summed Access Points Centrality'] = training_paths_df.apply(lambda x: return_edge_centrality_summation(x['access_points_edges'],edge_info_df),axis=1)
    
    # Get the occurences of road classes over the path
    road_classes_occurences_over_path_df= training_paths_df.apply(lambda x: return_aggregate_path_road_class_counts(edge_info_df,x['edges_passed'],'Path'),axis=1,result_type='expand')
    
    # Get the occurences of the road classes acting as access points
    road_classes_occurences_access_points_df = training_paths_df.apply(lambda x: return_aggregate_path_road_class_counts(edge_info_df,x['access_points_edges'],'Access Points'),axis=1,result_type='expand')
    
    # Get the centrality scores across road class for list of edges per path
    
    road_classes_centrality_scores_df = training_paths_df.apply(lambda x: return_centrality_scores_by_road_class(edge_info_df,x['access_points_edges']),axis=1,result_type='expand')
    
    # Get the source and target node features
    
    source_node_features_df = training_paths_df.apply(lambda x: return_node_features(x[NODES_PASSED_COLUMN_NAME][0],node_info_df,edge_info_df,prefix="Source"),axis=1,result_type='expand')
    target_node_features_df = training_paths_df.apply(lambda x: return_node_features(x[NODES_PASSED_COLUMN_NAME][-1],node_info_df,edge_info_df,prefix="Target"),axis=1,result_type='expand')
    
    # Create final df
    
    final_training_paths_df = pd.concat([training_paths_df,road_classes_occurences_access_points_df,road_classes_occurences_over_path_df,road_classes_centrality_scores_df,source_node_features_df,target_node_features_df],axis=1,)
    
    final_training_paths_df = final_training_paths_df.drop(labels=['edges_passed','nodes_passed','access_points_edges'],axis=1)
    return final_training_paths_df
    
        
if __name__ == "__main__":
    df = create_features()
    
    df.to_excel('./data/Features_node_information.xlsx',index=False)