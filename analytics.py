import sys
import pandas as pd
import numpy as np

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

def generate_insight_1(data: pd.DataFrame) -> str:
    """Generate insight about cattle milk yield dataset structure and target variable"""
    
    insight = "INSIGHT 1: CATTLE MILK YIELD DATASET OVERVIEW\n"
    insight += "=" * 70 + "\n\n"
    
    # Basic statistics
    insight += f"Dataset Dimensions:\n"
    insight += f"- Total Records (Cattle Observations): {data.shape[0]:,}\n"
    insight += f"- Total Features: {data.shape[1]}\n"
    insight += f"- Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
    
    # Target variable analysis
    if 'Milk_Yield_L' in data.columns:
        yield_data = data['Milk_Yield_L']
        insight += "Target Variable (Milk_Yield_L) Analysis:\n"
        insight += f"- Mean Milk Yield: {yield_data.mean():.2f} liters/day\n"
        insight += f"- Median Milk Yield: {yield_data.median():.2f} liters/day\n"
        insight += f"- Std Deviation: {yield_data.std():.2f} liters\n"
        insight += f"- Range: {yield_data.min():.2f} - {yield_data.max():.2f} liters\n"
        insight += f"- Coefficient of Variation: {(yield_data.std()/yield_data.mean())*100:.1f}%\n\n"
        
        # Quartile analysis
        q1 = yield_data.quantile(0.25)
        q2 = yield_data.quantile(0.50)
        q3 = yield_data.quantile(0.75)
        insight += f"Milk Yield Distribution Quartiles:\n"
        insight += f"- Q1 (25th percentile): {q1:.2f} liters\n"
        insight += f"- Q2 (50th percentile/Median): {q2:.2f} liters\n"
        insight += f"- Q3 (75th percentile): {q3:.2f} liters\n"
        insight += f"- Interquartile Range (IQR): {q3-q1:.2f} liters\n\n"
    
    # Feature categories breakdown
    encoded_cols = [col for col in data.columns if col.endswith('_encoded')]
    scaled_cols = [col for col in data.columns if col.endswith('_scaled')]
    pca_cols = [col for col in data.columns if col.startswith('PC_')]
    engineered_features = [col for col in data.columns if col in [
        'Total_Vaccinations', 'Weight_Age_Ratio', 'Yield_per_kg_Weight',
        'Activity_Score', 'Rest_Rumination_Balance'
    ]]
    
    insight += "Feature Categories:\n"
    insight += f"- Original Numeric Features: {len([c for c in NUMERIC_COLS if c in data.columns])}\n"
    insight += f"- Vaccine Features (Binary): {len([c for c in VACCINE_COLS if c in data.columns])}\n"
    insight += f"- Encoded Categorical Features: {len(encoded_cols)}\n"
    insight += f"- Scaled Features: {len(scaled_cols)}\n"
    insight += f"- PCA Components: {len(pca_cols)}\n"
    insight += f"- Engineered Features: {len(engineered_features)}\n\n"
    
    # Data quality
    missing_total = data.isnull().sum().sum()
    missing_percentage = (missing_total / (data.shape[0] * data.shape[1])) * 100
    insight += f"Data Quality:\n"
    insight += f"- Missing Values: {missing_total} ({missing_percentage:.2f}% of total data)\n"
    insight += f"- Complete Records: {data.dropna().shape[0]:,} out of {data.shape[0]:,} ({data.dropna().shape[0]/data.shape[0]*100:.1f}%)\n"
    
    return insight

def generate_insight_2(data: pd.DataFrame) -> str:
    """Generate insight about cattle health, management, and productivity factors"""
    
    insight = "INSIGHT 2: CATTLE HEALTH, MANAGEMENT & PRODUCTIVITY FACTORS\n"
    insight += "=" * 70 + "\n\n"
    
    # Vaccination coverage analysis
    vaccine_features = [col for col in VACCINE_COLS if col in data.columns]
    if vaccine_features:
        insight += "Vaccination Coverage Analysis:\n"
        
        if 'Total_Vaccinations' in data.columns:
            total_vacc = data['Total_Vaccinations']
            insight += f"- Average Vaccinations per Cattle: {total_vacc.mean():.2f} out of {len(vaccine_features)}\n"
            insight += f"- Cattle with Full Vaccination: {(total_vacc == len(vaccine_features)).sum():,} ({(total_vacc == len(vaccine_features)).sum()/len(data)*100:.1f}%)\n"
            insight += f"- Cattle with No Vaccination: {(total_vacc == 0).sum():,} ({(total_vacc == 0).sum()/len(data)*100:.1f}%)\n"
        
        # Individual vaccine coverage
        insight += "\nIndividual Vaccine Coverage:\n"
        for vaccine in vaccine_features[:5]:  # Show top 5
            coverage = (data[vaccine] == 1).sum()
            coverage_pct = (coverage / len(data)) * 100
            insight += f"- {vaccine.replace('_Vaccine', '')}: {coverage:,} cattle ({coverage_pct:.1f}%)\n"
        insight += "\n"
    
    # Cattle characteristics
    insight += "Cattle Physical Characteristics:\n"
    if 'Age_Months' in data.columns:
        age = data['Age_Months']
        insight += f"- Average Age: {age.mean():.1f} months ({age.mean()/12:.1f} years)\n"
        insight += f"- Age Range: {age.min():.0f} - {age.max():.0f} months\n"
    
    if 'Weight_kg' in data.columns:
        weight = data['Weight_kg']
        insight += f"- Average Weight: {weight.mean():.1f} kg\n"
        insight += f"- Weight Range: {weight.min():.0f} - {weight.max():.0f} kg\n"
    
    if 'Parity' in data.columns:
        parity = data['Parity']
        insight += f"- Average Parity (Calving Count): {parity.mean():.1f}\n"
        insight += f"- Parity Range: {parity.min():.0f} - {parity.max():.0f}\n"
    
    if 'Body_Condition_Score' in data.columns:
        bcs = data['Body_Condition_Score']
        insight += f"- Average Body Condition Score: {bcs.mean():.2f}\n"
        insight += f"- BCS Range: {bcs.min():.1f} - {bcs.max():.1f}\n"
    
    insight += "\n"
    
    # Management and feeding practices
    insight += "Management & Feeding Practices:\n"
    if 'Feed_Quantity_kg' in data.columns:
        feed = data['Feed_Quantity_kg']
        insight += f"- Average Feed Quantity: {feed.mean():.2f} kg/day\n"
        insight += f"- Feed Range: {feed.min():.1f} - {feed.max():.1f} kg\n"
    
    if 'Water_Intake_L' in data.columns:
        water = data['Water_Intake_L']
        insight += f"- Average Water Intake: {water.mean():.1f} liters/day\n"
        insight += f"- Water Range: {water.min():.0f} - {water.max():.0f} liters\n"
    
    if 'Feeding_Frequency' in data.columns:
        freq = data['Feeding_Frequency']
        insight += f"- Average Feeding Frequency: {freq.mean():.1f} times/day\n"
    
    if 'Housing_Score' in data.columns:
        housing = data['Housing_Score']
        insight += f"- Average Housing Score: {housing.mean():.2f}\n"
    
    insight += "\n"
    
    # Activity and behavior patterns
    insight += "Activity & Behavior Patterns:\n"
    if 'Walking_Distance_km' in data.columns:
        walking = data['Walking_Distance_km']
        insight += f"- Average Walking Distance: {walking.mean():.2f} km/day\n"
    
    if 'Grazing_Duration_hrs' in data.columns:
        grazing = data['Grazing_Duration_hrs']
        insight += f"- Average Grazing Duration: {grazing.mean():.1f} hours/day\n"
    
    if 'Rumination_Time_hrs' in data.columns:
        rumination = data['Rumination_Time_hrs']
        insight += f"- Average Rumination Time: {rumination.mean():.1f} hours/day\n"
    
    if 'Resting_Hours' in data.columns:
        resting = data['Resting_Hours']
        insight += f"- Average Resting Time: {resting.mean():.1f} hours/day\n"
    
    # Engineered features insights
    if 'Activity_Score' in data.columns:
        activity = data['Activity_Score']
        insight += f"\n- Combined Activity Score: {activity.mean():.2f} (normalized)\n"
    
    return insight

def generate_insight_3(data: pd.DataFrame) -> str:
    """Generate insight about milk yield patterns and key correlations"""
    
    insight = "INSIGHT 3: MILK YIELD PATTERNS & KEY CORRELATIONS\n"
    insight += "=" * 70 + "\n\n"
    
    # Milk yield patterns
    if 'Milk_Yield_L' in data.columns:
        yield_data = data['Milk_Yield_L']
        
        # Yield distribution analysis
        low_yield = (yield_data < yield_data.quantile(0.25)).sum()
        medium_yield = ((yield_data >= yield_data.quantile(0.25)) & (yield_data <= yield_data.quantile(0.75))).sum()
        high_yield = (yield_data > yield_data.quantile(0.75)).sum()
        
        insight += "Milk Yield Distribution Categories:\n"
        insight += f"- Low Yield (<Q1): {low_yield:,} cattle ({low_yield/len(data)*100:.1f}%)\n"
        insight += f"- Medium Yield (Q1-Q3): {medium_yield:,} cattle ({medium_yield/len(data)*100:.1f}%)\n"
        insight += f"- High Yield (>Q3): {high_yield:,} cattle ({high_yield/len(data)*100:.1f}%)\n\n"
        
        # Correlation analysis with key features
        insight += "Top Correlations with Milk Yield:\n"
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_data = data[numeric_cols]
        
        if 'Milk_Yield_L' in numeric_data.columns:
            correlations = numeric_data.corr()['Milk_Yield_L'].drop('Milk_Yield_L', errors='ignore')
            correlations = correlations.abs().sort_values(ascending=False)
            
            # Top positive and negative correlations
            top_n = min(10, len(correlations))
            insight += f"\nTop {top_n} Features by Absolute Correlation:\n"
            for i, (feature, corr_value) in enumerate(correlations.head(top_n).items(), 1):
                actual_corr = numeric_data.corr()['Milk_Yield_L'][feature]
                direction = "positive" if actual_corr > 0 else "negative"
                insight += f"{i}. {feature}: {abs(actual_corr):.3f} ({direction})\n"
        
        insight += "\n"
        
        # Environmental factors impact
        insight += "Environmental Factors Impact:\n"
        if 'Ambient_Temperature_C' in data.columns:
            temp = data['Ambient_Temperature_C']
            insight += f"- Average Temperature: {temp.mean():.1f}°C (Range: {temp.min():.1f} - {temp.max():.1f}°C)\n"
            
            # Temperature effect on yield
            if 'Milk_Yield_L' in data.columns:
                temp_yield_corr = data[['Ambient_Temperature_C', 'Milk_Yield_L']].corr().iloc[0, 1]
                insight += f"- Temperature-Yield Correlation: {temp_yield_corr:.3f}\n"
        
        if 'Humidity_percent' in data.columns:
            humidity = data['Humidity_percent']
            insight += f"- Average Humidity: {humidity.mean():.1f}% (Range: {humidity.min():.0f} - {humidity.max():.0f}%)\n"
            
            if 'Milk_Yield_L' in data.columns:
                humidity_yield_corr = data[['Humidity_percent', 'Milk_Yield_L']].corr().iloc[0, 1]
                insight += f"- Humidity-Yield Correlation: {humidity_yield_corr:.3f}\n"
        
        insight += "\n"
        
        # Lactation stage and yield
        if 'Days_in_Milk' in data.columns:
            dim = data['Days_in_Milk']
            insight += "Lactation Analysis:\n"
            insight += f"- Average Days in Milk: {dim.mean():.0f} days\n"
            insight += f"- Range: {dim.min():.0f} - {dim.max():.0f} days\n"
            
            if 'Milk_Yield_L' in data.columns:
                dim_yield_corr = data[['Days_in_Milk', 'Milk_Yield_L']].corr().iloc[0, 1]
                insight += f"- Days in Milk vs Yield Correlation: {dim_yield_corr:.3f}\n"
        
        # Previous yield predictor
        if 'Previous_Week_Avg_Yield' in data.columns and 'Milk_Yield_L' in data.columns:
            prev_yield_corr = data[['Previous_Week_Avg_Yield', 'Milk_Yield_L']].corr().iloc[0, 1]
            insight += f"\nPrevious Week Yield Predictive Power:\n"
            insight += f"- Correlation with Current Yield: {prev_yield_corr:.3f}\n"
            insight += f"- This indicates {'strong' if abs(prev_yield_corr) > 0.7 else 'moderate' if abs(prev_yield_corr) > 0.4 else 'weak'} temporal consistency\n"
        
        # PCA insights
        pca_cols = [col for col in data.columns if col.startswith('PC_')]
        if pca_cols and 'Milk_Yield_L' in data.columns:
            insight += f"\nPCA Components Relationship with Yield:\n"
            for pc in pca_cols[:3]:  # Top 3 components
                pc_yield_corr = data[[pc, 'Milk_Yield_L']].corr().iloc[0, 1]
                insight += f"- {pc} vs Yield Correlation: {pc_yield_corr:.3f}\n"
    
    return insight

def main():
    """
    Main analytics function that generates three domain-specific textual insights
    for cattle milk yield prediction dataset
    """
    # Get input file from command line argument
    if len(sys.argv) < 2:
        print("Error: Please provide input CSV file path as argument")
        print("Usage: python analytics.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Load the preprocessed data
        print(f"Loading preprocessed data from: {input_file}")
        data = pd.read_csv(input_file)
        print(f"Data shape: {data.shape}")
        
        # Generate insights
        print("\nGenerating Insight 1: Dataset Overview & Target Analysis...")
        insight1 = generate_insight_1(data)
        
        print("Generating Insight 2: Cattle Health & Management Factors...")
        insight2 = generate_insight_2(data)
        
        print("Generating Insight 3: Milk Yield Patterns & Correlations...")
        insight3 = generate_insight_3(data)
        
        # Save insights to separate files
        with open("insight1.txt", "w") as f:
            f.write(insight1)
        print("✅ Saved insight1.txt")
        
        with open("insight2.txt", "w") as f:
            f.write(insight2)
        print("✅ Saved insight2.txt")
        
        with open("insight3.txt", "w") as f:
            f.write(insight3)
        print("✅ Saved insight3.txt")
        
        # Print summary
        print(f"\n Analytics Summary:")
        print(f"- Generated 3 domain-specific textual insights")
        print(f"- Input data: {data.shape[0]:,} rows, {data.shape[1]} columns")
        print(f"- Output files: insight1.txt, insight2.txt, insight3.txt")
        print(f"- Focus: Cattle milk yield prediction analysis")
        
        # Call next script in pipeline
        print(f"\n Calling next script: visualize.py")
        import subprocess
        subprocess.run(["python", "visualize.py", input_file])
        
    except Exception as e:
        print(f"❌ Error during analytics generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()