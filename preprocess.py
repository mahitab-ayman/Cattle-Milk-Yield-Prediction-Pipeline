import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings


# Dataset-specific column definitions
IDENTIFIER_COLS = ['Cattle_ID', 'Farm_ID', 'Date']


CATEGORICAL_COLS = [
    'Breed', 'Region', 'Country', 'Climate_Zone', 
    'Management_System', 'Lactation_Stage', 'Feed_Type', 'Season'
]


VACCINE_COLS = [
    'FMD_Vaccine', 'Brucellosis_Vaccine', 'HS_Vaccine', 'BQ_Vaccine',
    'Anthrax_Vaccine', 'IBR_Vaccine', 'BVD_Vaccine', 'Rabies_Vaccine'
]


NUMERIC_COLS = [
    'Age_Months', 'Weight_kg', 'Parity', 'Days_in_Milk', 
    'Feed_Quantity_kg', 'Feeding_Frequency', 'Water_Intake_L',
    'Walking_Distance_km', 'Grazing_Duration_hrs', 'Rumination_Time_hrs',
    'Resting_Hours', 'Ambient_Temperature_C', 'Humidity_percent',
    'Housing_Score', 'Previous_Week_Avg_Yield', 'Body_Condition_Score',
    'Milking_Interval_hrs', 'Milk_Yield_L'
]


def main():
    """
    Main preprocessing function for Docker pipeline
    Takes input from ingest.py and prepares data for analytics
    Specifically optimized for cattle milk yield prediction dataset
    """
    # Get input file from command line argument
    if len(sys.argv) < 2:
        print("Error: Please provide input CSV file path as argument")
        print("Usage: python preprocess.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Load the raw data
        print(f"Loading data from: {input_file}")
        data = pd.read_csv(input_file)
        print(f"Original data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Create a copy to preserve original data structure
        processed_data = data.copy()
        
        # ============================================================================
        # STAGE 1: DATA CLEANING
        # ============================================================================
        print("\n=== STAGE 1: DATA CLEANING ===")
        
        # Task 1: Handle missing values
        print("1. Handling missing values...")
        missing_before = processed_data.isnull().sum().sum()
        print(f"   Total missing values before: {missing_before}")
        
        # Handle numeric columns (excluding target variable Milk_Yield_L for imputation)
        numeric_features = [col for col in NUMERIC_COLS if col in processed_data.columns and col != 'Milk_Yield_L']
        for col in numeric_features:
            if processed_data[col].isnull().sum() > 0:
                median_val = processed_data[col].median()
                processed_data[col].fillna(median_val, inplace=True)
                print(f"   ✓ Filled missing values in '{col}' with median: {median_val:.2f}")
        
        # Handle vaccine columns (binary) - fill with 0 (not vaccinated)
        vaccine_features = [col for col in VACCINE_COLS if col in processed_data.columns]
        for col in vaccine_features:
            if processed_data[col].isnull().sum() > 0:
                processed_data[col].fillna(0, inplace=True)
                print(f"   ✓ Filled missing values in '{col}' with 0 (not vaccinated)")
        
        # Handle categorical columns
        categorical_features = [col for col in CATEGORICAL_COLS if col in processed_data.columns]
        for col in categorical_features:
            if processed_data[col].isnull().sum() > 0:
                mode_val = processed_data[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                processed_data[col].fillna(fill_value, inplace=True)
                print(f"   ✓ Filled missing values in '{col}' with mode: '{fill_value}'")
        
        # Handle target variable (Milk_Yield_L) - drop rows with missing target
        if 'Milk_Yield_L' in processed_data.columns:
            missing_target = processed_data['Milk_Yield_L'].isnull().sum()
            if missing_target > 0:
                processed_data = processed_data.dropna(subset=['Milk_Yield_L'])
                print(f"   ✓ Dropped {missing_target} rows with missing target variable (Milk_Yield_L)")
        
        missing_after = processed_data.isnull().sum().sum()
        print(f"   Total missing values after: {missing_after}")
        
        # Task 2: Remove duplicates
        print("\n2. Removing duplicates...")
        duplicates_before = processed_data.duplicated().sum()
        print(f"   Duplicate rows found: {duplicates_before}")
        
        if duplicates_before > 0:
            processed_data = processed_data.drop_duplicates()
            print(f"   ✓ Removed {duplicates_before} duplicate rows")
        else:
            print("   ✓ No duplicates found")
        
        print(f"   Data shape after cleaning: {processed_data.shape}")
        
        # ============================================================================
        # STAGE 2: FEATURE TRANSFORMATION  
        # ============================================================================
        print("\n=== STAGE 2: FEATURE TRANSFORMATION ===")
        
        # Task 1: Encode categorical columns
        print("1. Encoding categorical columns...")
        
        encoded_count = 0
        for col in categorical_features:
            try:
                unique_count = processed_data[col].nunique()
                if unique_count <= 50:  # Reasonable threshold
                    le = LabelEncoder()
                    processed_data[f"{col}_encoded"] = le.fit_transform(processed_data[col].astype(str))
                    encoded_count += 1
                    print(f"   ✓ Encoded '{col}' ({unique_count} categories)")
                else:
                    print(f"     Skipped '{col}' (too many categories: {unique_count})")
            except Exception as e:
                print(f"   ✗ Failed to encode '{col}': {e}")
        
        print(f"   Successfully encoded {encoded_count} categorical columns")
        
        # Task 1b: Ensure vaccine columns are numeric (0/1)
        print("\n1b. Processing vaccine columns as binary features...")
        for col in vaccine_features:
            try:
                # Convert to numeric if not already
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0).astype(int)
                print(f"   ✓ Converted '{col}' to binary (0/1)")
            except Exception as e:
                print(f"   ✗ Failed to process '{col}': {e}")
        
        # Task 2: Scale numeric columns (excluding identifiers and target)
        print("\n2. Scaling numeric columns...")
        
        # Get all numeric columns to scale (exclude target variable for now)
        scale_cols = [col for col in numeric_features if col in processed_data.columns]
        
        if len(scale_cols) == 0:
            print("     No numeric columns found to scale")
        else:
            scaler = StandardScaler()
            scaled_count = 0
            
            for col in scale_cols:
                try:
                    # Skip if constant or has too few unique values
                    if processed_data[col].nunique() > 1:
                        processed_data[f"{col}_scaled"] = scaler.fit_transform(processed_data[[col]])
                        scaled_count += 1
                        print(f"   ✓ Scaled '{col}'")
                    else:
                        print(f"     Skipped '{col}' (constant values)")
                except Exception as e:
                    print(f"   ✗ Failed to scale '{col}': {e}")
            
            print(f"   Successfully scaled {scaled_count} numeric columns")
        
        # ============================================================================
        # STAGE 3: DIMENSIONALITY REDUCTION (PCA)
        # ============================================================================
        print("\n=== STAGE 3: DIMENSIONALITY REDUCTION ===")
        
        # Task: Apply PCA on scaled numeric features
        print("Applying Principal Component Analysis (PCA)...")
        
        # Get all scaled numeric columns
        scaled_cols = [col for col in processed_data.columns if col.endswith('_scaled')]
        
        if len(scaled_cols) < 2:
            print("     Insufficient scaled features for PCA (need at least 2)")
            print("   Continuing without PCA...")
        else:
            try:
                # Prepare data for PCA
                pca_data = processed_data[scaled_cols].dropna()
                
                if len(pca_data) < 2:
                    print("     Not enough data points for PCA after removing NaN")
                else:
                    # Determine optimal number of components
                    n_components = min(5, len(scaled_cols), len(pca_data) - 1)
                    
                    # Apply PCA
                    pca = PCA(n_components=n_components, random_state=42)
                    pca_components = pca.fit_transform(pca_data)
                    
                    # Add PCA components to dataframe
                    for i in range(n_components):
                        col_name = f'PC_{i+1}'
                        # Initialize with NaN
                        processed_data[col_name] = np.nan
                        # Fill with PCA values where data was available
                        processed_data.loc[pca_data.index, col_name] = pca_components[:, i]
                    
                    # Print PCA results
                    print(f"   ✓ PCA applied successfully")
                    print(f"   Input features: {len(scaled_cols)}")
                    print(f"   PCA components: {n_components}")
                    print(f"   Explained variance ratio:")
                    for i, variance in enumerate(pca.explained_variance_ratio_, 1):
                        print(f"     PC{i}: {variance:.4f} ({variance*100:.2f}%)")
                    total_variance = sum(pca.explained_variance_ratio_)
                    print(f"   Total variance explained: {total_variance:.4f} ({total_variance*100:.2f}%)")
                    
            except Exception as e:
                print(f"   ✗ PCA failed: {e}")
                print("   Continuing without PCA...")
        
        # ============================================================================
        # STAGE 4: DISCRETIZATION (BINNING) - REQUIRED BY ASSIGNMENT
        # ============================================================================
        print("\n=== STAGE 4: DISCRETIZATION ===")
        print("Binning numeric columns into categorical ranges...")
        
        # Task 1: Discretize Age_Months into age groups
        if 'Age_Months' in processed_data.columns:
            bins = [0, 24, 48, 72, 120, float('inf')]
            labels = ['Young (<2yr)', 'Adult (2-4yr)', 'Mature (4-6yr)', 'Senior (6-10yr)', 'Very Senior (>10yr)']
            processed_data['Age_Category'] = pd.cut(
                processed_data['Age_Months'], 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            )
            print(f"   ✓ Created 'Age_Category' with {len(labels)} bins")
            print(f"     Distribution: {processed_data['Age_Category'].value_counts().to_dict()}")
        
        # Task 2: Discretize Milk_Yield_L into productivity categories
        if 'Milk_Yield_L' in processed_data.columns:
            # Use quantile-based binning for more balanced categories
            processed_data['Yield_Category'] = pd.qcut(
                processed_data['Milk_Yield_L'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            print(f"   ✓ Created 'Yield_Category' with 4 quantile-based bins")
            print(f"     Distribution: {processed_data['Yield_Category'].value_counts().to_dict()}")
        
        # Task 3: Discretize Body_Condition_Score into health categories
        if 'Body_Condition_Score' in processed_data.columns:
            bins = [0, 2.5, 3.5, 4.5, 5.0]
            labels = ['Poor', 'Fair', 'Good', 'Excellent']
            processed_data['BCS_Category'] = pd.cut(
                processed_data['Body_Condition_Score'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            print(f"   ✓ Created 'BCS_Category' with {len(labels)} bins")
            print(f"     Distribution: {processed_data['BCS_Category'].value_counts().to_dict()}")
        
        # Encode the discretized categorical columns for modeling
        print("\nEncoding discretized features...")
        discretized_cols = ['Age_Category', 'Yield_Category', 'BCS_Category']
        for col in discretized_cols:
            if col in processed_data.columns:
                try:
                    le = LabelEncoder()
                    processed_data[f"{col}_encoded"] = le.fit_transform(processed_data[col].astype(str))
                    print(f"   ✓ Encoded '{col}'")
                except Exception as e:
                    print(f"   ✗ Failed to encode '{col}': {e}")
        
        # ============================================================================
        # STAGE 5: FEATURE ENGINEERING (Domain-specific)
        # ============================================================================
        print("\n=== STAGE 5: FEATURE ENGINEERING ===")
        
        print("Creating domain-specific features for cattle milk yield prediction...")
        
        # Create total vaccination score
        if len(vaccine_features) > 0:
            processed_data['Total_Vaccinations'] = processed_data[vaccine_features].sum(axis=1)
            print(f"   ✓ Created 'Total_Vaccinations' (sum of {len(vaccine_features)} vaccines)")
        
        # Create age-weight ratio (if both exist)
        if 'Age_Months' in processed_data.columns and 'Weight_kg' in processed_data.columns:
            processed_data['Weight_Age_Ratio'] = processed_data['Weight_kg'] / (processed_data['Age_Months'] + 1)
            print("   ✓ Created 'Weight_Age_Ratio' feature")
        
        # Create daily milk per kg body weight (productivity indicator)
        if 'Milk_Yield_L' in processed_data.columns and 'Weight_kg' in processed_data.columns:
            processed_data['Yield_per_kg_Weight'] = processed_data['Milk_Yield_L'] / (processed_data['Weight_kg'] + 1)
            print("   ✓ Created 'Yield_per_kg_Weight' feature")
        
        # Create activity score (combination of walking and grazing)
        if 'Walking_Distance_km' in processed_data.columns and 'Grazing_Duration_hrs' in processed_data.columns:
            processed_data['Activity_Score'] = (
                processed_data['Walking_Distance_km'] * 0.5 + 
                processed_data['Grazing_Duration_hrs'] * 0.5
            )
            print("   ✓ Created 'Activity_Score' feature")
        
        # Create rest-rumination balance
        if 'Resting_Hours' in processed_data.columns and 'Rumination_Time_hrs' in processed_data.columns:
            processed_data['Rest_Rumination_Balance'] = (
                processed_data['Resting_Hours'] / (processed_data['Rumination_Time_hrs'] + 1)
            )
            print("   ✓ Created 'Rest_Rumination_Balance' feature")
        
        # ============================================================================
        # FINAL DATA PREPARATION
        # ============================================================================
        print("\n=== FINAL DATA PREPARATION ===")
        
        # Select final features for output
        final_columns = []
        
        # Keep identifier columns (for tracking)
        identifier_cols = [col for col in IDENTIFIER_COLS if col in processed_data.columns]
        final_columns.extend(identifier_cols)
        
        # Keep original numeric columns (including target)
        orig_numeric = [col for col in NUMERIC_COLS if col in processed_data.columns]
        final_columns.extend(orig_numeric)
        
        # Keep vaccine columns
        final_columns.extend(vaccine_features)
        
        # Keep encoded categorical columns
        encoded_cols = [col for col in processed_data.columns if col.endswith('_encoded')]
        final_columns.extend(encoded_cols)
        
        # Keep scaled columns  
        final_columns.extend(scaled_cols)
        
        # Keep PCA components
        pca_cols = [col for col in processed_data.columns if col.startswith('PC_')]
        final_columns.extend(pca_cols)
        
        # Keep engineered features
        engineered_features = [
            'Total_Vaccinations', 'Weight_Age_Ratio', 'Yield_per_kg_Weight',
            'Activity_Score', 'Rest_Rumination_Balance'
        ]
        for feat in engineered_features:
            if feat in processed_data.columns:
                final_columns.extend([feat])
        
        # Keep discretized features
        discretized_features = [
            'Age_Category', 'Yield_Category', 'BCS_Category',
            'Age_Category_encoded', 'Yield_Category_encoded', 'BCS_Category_encoded'
        ]
        for feat in discretized_features:
            if feat in processed_data.columns:
                final_columns.append(feat)
        
        # Ensure all columns exist and create final dataset
        existing_columns = [col for col in final_columns if col in processed_data.columns]
        final_data = processed_data[existing_columns].copy()
        
        # Remove any remaining NaN values (if any)
        rows_before = len(final_data)
        final_data = final_data.dropna()
        rows_dropped = rows_before - len(final_data)
        if rows_dropped > 0:
            print(f"     Dropped {rows_dropped} rows with NaN values")
        
        print(f"\nFinal dataset shape: {final_data.shape}")
        print(f"Final columns: {len(final_data.columns)}")
        print(f"Column types:")
        for dtype, count in final_data.dtypes.value_counts().items():
            print(f"  {dtype}: {count}")
        
        # Print summary statistics
        print("\n=== DATA SUMMARY ===")
        if 'Milk_Yield_L' in final_data.columns:
            print(f"Target variable (Milk_Yield_L) statistics:")
            print(f"  Mean: {final_data['Milk_Yield_L'].mean():.2f} L")
            print(f"  Median: {final_data['Milk_Yield_L'].median():.2f} L")
            print(f"  Std Dev: {final_data['Milk_Yield_L'].std():.2f} L")
            print(f"  Min: {final_data['Milk_Yield_L'].min():.2f} L")
            print(f"  Max: {final_data['Milk_Yield_L'].max():.2f} L")
        
        # Save processed data
        output_file = "data_preprocessed.csv"
        final_data.to_csv(output_file, index=False)
        print(f"\n Preprocessed data saved as: {output_file}")
        
        # Call next script in pipeline
        print(f"\n Calling next script: analytics.py")
        import subprocess
        subprocess.run(["python", "analytics.py", output_file])
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
