import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.spatial import cKDTree
import hydroeval as he

def evaluate_regression_model(y_true, y_pred):
    """Evaluate regression model performance metrics."""
    nse = he.evaluator(he.nse, y_pred, y_true)
    kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_true)
    mse = metrics.mean_absolute_error(y_true, y_pred)
    r_squared = ((sum((y_true - y_true.mean()) * (y_pred - y_pred.mean()))) /
                 ((np.sqrt(sum((y_true - y_true.mean())**2))) *
                  (np.sqrt(sum((y_pred - y_pred.mean())**2)))))**2
    return {'NSE': nse, 'KGE': kge, 'MSE': mse, 'R_Squared': r_squared}

def save_metrics(metrics_dict, output_file):
    """Save model evaluation metrics to a CSV file."""
    df_metrics = pd.DataFrame([metrics_dict])
    df_metrics.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")

def preprocess_spatial_data(spatial_file, original_file, scenario, cluster, year, output_dir):
    """Preprocess spatial data for regression."""
    df_spatial = pd.read_csv(spatial_file)
    df_orig = pd.read_csv(original_file)

    # Drop unnecessary columns and add geographic coordinates
    df_spatial = df_spatial.drop(columns=[
        'SLR_predicted', 'OC_USA_EBK', 'GH_USA_EBK', 'SALT_USA_EBK', 
        'MSLP_USA_EBK', 'SP_USA_EBK', 'SST_USA_EBK', 'VLM_shcua_EBK'
    ])
    df_spatial['Latitude1_rad'] = np.radians(df_spatial['Y'])
    df_spatial['Longitude1_rad'] = np.radians(df_spatial['X'])
    df_orig['Latitude1_rad'] = np.radians(df_orig['Y_orig'])
    df_orig['Longitude1_rad'] = np.radians(df_orig['X_orig'])

    # Build KDTree for spatial proximity
    tree = cKDTree(df_orig[['Latitude1_rad', 'Longitude1_rad']].values)
    distances, indices = tree.query(df_spatial[['Latitude1_rad', 'Longitude1_rad']].values, k=2)

    # Assign closest point distances and indices
    df_spatial['first_point_distance'] = distances[:, 0]
    df_spatial['second_point_distance'] = distances[:, 1]
    df_spatial['first_closest_point_index'] = indices[:, 0]
    df_spatial['second_closest_point_index'] = indices[:, 1]

    # Assign rates for different years
    for target_year in ['2020', '2030', '2050', '2100']:
        df_spatial[f'first_point_{target_year}_rate'] = df_orig.iloc[
            df_spatial['first_closest_point_index'], df_orig.columns.get_loc(f'{target_year}_Rates')].values
        df_spatial[f'second_point_{target_year}_rate'] = df_orig.iloc[
            df_spatial['second_closest_point_index'], df_orig.columns.get_loc(f'{target_year}_Rates')].values

    # Compute alpha values
    df_spatial['alpha_2020_first_point'] = df_spatial.iloc[:, 2] / df_spatial['first_point_2020_rate']
    df_spatial['alpha_2020_second_point'] = df_spatial.iloc[:, 2] / df_spatial['second_point_2020_rate']

    # Clean up and save preprocessed data
    df_spatial = df_spatial.replace([np.inf, -np.inf, np.nan], 0)
    output_file = os.path.join(output_dir, f'input_features_{year}_{scenario}_{cluster}.csv')
    df_spatial.to_csv(output_file, index=False)
    print(f"Preprocessed spatial data saved to {output_file}")
    return df_spatial

def predict_values(df_spatial, model, year, scenario, cluster, output_dir):
    """Predict values using regression model."""
    X_unlabeled = df_spatial[[f'first_point_{year}_rate', f'second_point_{year}_rate', 
                              'alpha_2020_first_point', 'alpha_2020_second_point', 
                              'first_point_distance', 'second_point_distance']]
    df_spatial[f'predicted_{year}'] = model.predict(X_unlabeled)

    # Save predictions
    output_file = os.path.join(output_dir, f'predicted_values_{year}_{scenario}_{cluster}.csv')
    df_spatial.to_csv(output_file, index=False)
    print(f"Predicted values saved to {output_file}")

if __name__ == "__main__":
    # Define scenarios, years, and clusters
    scenarios = ['2_45', '5_85']
    years = ['2020', '2030', '2050', '2100']
    clusters = ['second', 'third']
    base_output_dir = 'output_directory'

    for scenario in scenarios:
        for year in years:
            # Load original dataset
            original_file = f'data/USA_SLR_all_{scenario}_ready_for_regression.csv'
            df_orig = pd.read_csv(original_file)

            # Perform regression
            y = df_orig[f'Median_{year}']
            X = df_orig[[f'first_point_{year}_rate', f'second_point_{year}_rate', 
                         'alpha_2020_first_point', 'alpha_2020_second_point', 
                         'first_point_distance', 'second_point_distance']]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            # Evaluate model
            metrics_dict = evaluate_regression_model(y, y_pred)
            metrics_output = os.path.join(base_output_dir, f'metrics_{year}_{scenario}.csv')
            save_metrics(metrics_dict, metrics_output)

            # Process spatial data
            for cluster in clusters:
                spatial_file = f'data/SLR_spatial_{scenario}_{cluster}_cluster.csv'
                df_spatial = preprocess_spatial_data(spatial_file, original_file, scenario, cluster, year, base_output_dir)
                predict_values(df_spatial, model, year, scenario, cluster, base_output_dir)
