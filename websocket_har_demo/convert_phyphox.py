import os
import pandas as pd
import glob

# --- Configuration ---
SOURCE_DIR = "."  # Where your Phyphox CSVs are
TARGET_DIR = "data"
os.makedirs(TARGET_DIR, exist_ok=True)

# Mapping of file patterns to activities
# Example: if your file is 'walking_session_1.csv', it will be mapped to 'walking'
ACTIVITIES = ["walking", "running", "sitting", "standing"]

def convert():
    print(f"Searching for Phyphox CSVs in {SOURCE_DIR}...")
    
    # Phyphox "Accelerometer + Gyroscope" export typically looks like this:
    # Time (s), Acceleration x (m/s^2), Acceleration y (m/s^2), Acceleration z (m/s^2), Gyroscope x (rad/s), Gyroscope y (rad/s), Gyroscope z (rad/s)
    
    for activity in ACTIVITIES:
        # Look for files containing the activity name
        files = glob.glob(os.path.join(SOURCE_DIR, f"*{activity}*.csv"))
        
        if not files:
            continue
            
        print(f"Converting files for: {activity}")
        all_data = []
        
        for f in files:
            # Skip files already in the 'data' directory
            if TARGET_DIR in f:
                continue
                
            print(f"  Reading {f}...")
            try:
                # Phyphox uses semicolon by default in some regions, comma in others. 
                # We'll try to detect or just use read_csv defaults.
                df = pd.read_csv(f)
                
                # Normalize column names (Phyphox names are verbose)
                # We need: ts, ax, ay, az, gx, gy, gz
                
                # Typical Phyphox columns:
                # 0: Time (s)
                # 1: Acceleration x (m/s^2)
                # 2: Acceleration y (m/s^2)
                # 3: Acceleration z (m/s^2)
                # 4: Gyroscope x (rad/s)
                # 5: Gyroscope y (rad/s)
                # 6: Gyroscope z (rad/s)
                
                if len(df.columns) < 7:
                    print(f"    Warning: {f} has fewer than 7 columns. Skipping.")
                    continue
                
                # Create a new dataframe with our internal standard
                new_df = pd.DataFrame()
                new_df['ts'] = df.iloc[:, 0] * 1000 # Convert to ms
                new_df['ax'] = df.iloc[:, 1]
                new_df['ay'] = df.iloc[:, 2]
                new_df['az'] = df.iloc[:, 3]
                new_df['gx'] = df.iloc[:, 4]
                new_df['gy'] = df.iloc[:, 5]
                new_df['gz'] = df.iloc[:, 6]
                
                all_data.append(new_df)
            except Exception as e:
                print(f"    Error reading {f}: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            output_path = os.path.join(TARGET_DIR, f"{activity}.csv")
            combined.to_csv(output_path, index=False)
            print(f"  Saved {len(combined)} rows to {output_path}")

if __name__ == "__main__":
    convert()
    print("\nConversion complete. You can now run: python train_demo_model.py")
