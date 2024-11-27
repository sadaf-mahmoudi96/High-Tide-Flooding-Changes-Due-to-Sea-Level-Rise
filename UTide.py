import os
import glob
import pandas as pd
from utide import solve, reconstruct
from matplotlib.dates import date2num

def generate_datetime_csv(start_date, end_date, output_file, freq='h'):
    """
    Generate a CSV file of datetime range with hourly intervals.
    """
    times = pd.date_range(start=start_date, end=end_date, freq=freq)
    times_df = pd.DataFrame(times)
    times_df.to_csv(output_file, index=False, header=False)
    print(f"Datetime CSV generated: {output_file}")

def process_tide_data(sail_file, path_to_results, dt_reconstruct_file):
    """
    Process tide data for multiple locations using UTide to generate projections.
    """
    sail = pd.read_csv(sail_file)
    file_list = glob.glob(path_to_results + "/*")

    for i in range(len(sail)):
        lat = sail.iloc[i, 1]
        location_file = sail.iloc[i, 2]

        # Find the matching result file
        for k in range(len(file_list)):
            if os.path.basename(file_list[k]) == location_file:
                print(f"Processing: {file_list[k]}")
                df = pd.read_csv(os.path.join(file_list[k], "Total_WL_hourly_biased_corrected.csv"))

                # Convert time and solve for tidal constituents
                df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
                time = date2num(df['Time'])
                coef = solve(time, df['biased_corrected'].values,
                             lat=lat, nodal=False, trend=True,
                             method='ols', conf_int='linear', Rayleigh_min=0.95)

                # Save tidal constituents
                coef_df = pd.DataFrame({
                    'name': coef.name,
                    'A': coef.A,
                    'A_interval': coef.A_ci,
                    'phase': coef.g,
                    'phase_interval': coef.g_ci,
                    'SNR': coef.SNR,
                    'Freq': coef.aux.frq
                })
                coef_df.to_csv(os.path.join(file_list[k], 'coef.csv'), index=False)
                print(f"Tidal constituents saved for {location_file}")

                # Reconstruct tide levels
                t_recons = pd.read_csv(dt_reconstruct_file, header=None, names=['Time'])
                t_recons['Time'] = pd.to_datetime(t_recons['Time'], format='%Y/%m/%d %H:%M:%S')
                time_tide = date2num(t_recons['Time'])
                tide = reconstruct(time_tide, coef, verbose=False)

                tide_df = pd.DataFrame({'Time': t_recons['Time'], 'Predicted': tide.h})
                tide_df.to_csv(os.path.join(file_list[k], 'tide_prediction.csv'), index=False)
                print(f"Tide prediction saved for {location_file}")
    print("Processing complete for all locations.")

# Example usage
if __name__ == "__main__":
    # Generate datetime CSV for tide reconstruction
    generate_datetime_csv(
        start_date='2021-01-01',
        end_date='2110-01-01',
        output_file='dt_reconstruct_true.csv'
    )

    # Process tide data
    process_tide_data(
        sail_file='sail_data.csv',
        path_to_results='results_directory',
        dt_reconstruct_file='dt_reconstruct_true.csv'
    )
