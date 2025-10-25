import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Dataset-specific column definitions
VACCINE_COLS = [
    'FMD_Vaccine', 'Brucellosis_Vaccine', 'HS_Vaccine', 'BQ_Vaccine',
    'Anthrax_Vaccine', 'IBR_Vaccine', 'BVD_Vaccine', 'Rabies_Vaccine'
]

def setup_plot_style():
    """Configure consistent plot styling for cattle milk yield visualizations"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def create_cattle_milk_yield_plot(data: pd.DataFrame) -> bool:
    """
    Create comprehensive visualization for cattle milk yield prediction dataset
    Returns True if successful, False otherwise
    """
    try:
        setup_plot_style()
        
        print(f"📊 Creating cattle milk yield visualization...")
        
        # Check if target variable exists
        has_target = 'Milk_Yield_L' in data.columns
        
        # Create a comprehensive 6-panel figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ====================================================================
        # PLOT 1: Milk Yield Distribution (Top Left - spans 2 columns)
        # ====================================================================
        if has_target:
            ax1 = fig.add_subplot(gs[0, :2])
            milk_yield = data['Milk_Yield_L']
            
            # Histogram with KDE
            sns.histplot(milk_yield, kde=True, bins=50, color='steelblue', ax=ax1)
            ax1.axvline(milk_yield.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {milk_yield.mean():.2f}L')
            ax1.axvline(milk_yield.median(), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {milk_yield.median():.2f}L')
            
            ax1.set_title('Milk Yield Distribution (Target Variable)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Milk Yield (Liters/Day)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.text(0.5, 0.5, 'Target Variable (Milk_Yield_L) Not Found', 
                    ha='center', va='center', fontsize=14)
            ax1.set_title('Milk Yield Distribution')
        
        # ====================================================================
        # PLOT 2: Vaccination Coverage (Top Right)
        # ====================================================================
        ax2 = fig.add_subplot(gs[0, 2])
        vaccine_cols_present = [col for col in VACCINE_COLS if col in data.columns]
        
        if vaccine_cols_present:
            vaccine_coverage = [(col.replace('_Vaccine', ''), 
                               (data[col] == 1).sum() / len(data) * 100) 
                              for col in vaccine_cols_present]
            vaccine_names, coverage_pct = zip(*vaccine_coverage)
            
            colors = ['green' if pct > 50 else 'orange' if pct > 25 else 'red' 
                     for pct in coverage_pct]
            
            ax2.barh(vaccine_names, coverage_pct, color=colors)
            ax2.set_xlabel('Coverage (%)')
            ax2.set_title('Vaccination Coverage', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No Vaccine Data', ha='center', va='center')
            ax2.set_title('Vaccination Coverage')
        
        # ====================================================================
        # PLOT 3: Top Correlations with Milk Yield (Middle Left)
        # ====================================================================
        ax3 = fig.add_subplot(gs[1, 0])
        
        if has_target:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlations = data[numeric_cols].corr()['Milk_Yield_L'].drop('Milk_Yield_L', errors='ignore')
            correlations = correlations.abs().sort_values(ascending=False).head(10)
            
            # Get actual correlation values (with sign)
            actual_corrs = data[numeric_cols].corr()['Milk_Yield_L'][correlations.index]
            colors = ['green' if x > 0 else 'red' for x in actual_corrs]
            
            ax3.barh(range(len(actual_corrs)), actual_corrs.values, color=colors)
            ax3.set_yticks(range(len(actual_corrs)))
            ax3.set_yticklabels([label[:20] for label in actual_corrs.index], fontsize=9)
            ax3.set_xlabel('Correlation Coefficient')
            ax3.set_title('Top 10 Features Correlated with Milk Yield', fontweight='bold')
            ax3.axvline(0, color='black', linestyle='-', linewidth=0.8)
            ax3.grid(True, alpha=0.3, axis='x')
        else:
            ax3.text(0.5, 0.5, 'Target Variable Required', ha='center', va='center')
            ax3.set_title('Correlations with Milk Yield')
        
        # ====================================================================
        # PLOT 4: Cattle Physical Characteristics (Middle Center)
        # ====================================================================
        ax4 = fig.add_subplot(gs[1, 1])
        
        physical_features = {
            'Age_Months': 'Age (months)',
            'Weight_kg': 'Weight (kg)',
            'Body_Condition_Score': 'Body Condition',
            'Parity': 'Parity (calvings)'
        }
        
        available_features = {k: v for k, v in physical_features.items() if k in data.columns}
        
        if available_features:
            feature_data = []
            feature_labels = []
            
            for col, label in available_features.items():
                feature_data.append(data[col].dropna())
                feature_labels.append(label)
            
            bp = ax4.boxplot(feature_data, labels=feature_labels, patch_artist=True)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax4.set_title('Cattle Physical Characteristics', fontweight='bold')
            ax4.set_ylabel('Value (Normalized Scale)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Physical Characteristics Not Found', 
                    ha='center', va='center')
            ax4.set_title('Cattle Physical Characteristics')
        
        # ====================================================================
        # PLOT 5: Environmental Factors vs Milk Yield (Middle Right)
        # ====================================================================
        ax5 = fig.add_subplot(gs[1, 2])
        
        if has_target and 'Ambient_Temperature_C' in data.columns:
            # Scatter plot with color gradient based on humidity if available
            if 'Humidity_percent' in data.columns:
                scatter = ax5.scatter(data['Ambient_Temperature_C'], 
                                     data['Milk_Yield_L'],
                                     c=data['Humidity_percent'], 
                                     cmap='viridis', 
                                     alpha=0.6, 
                                     s=20)
                plt.colorbar(scatter, ax=ax5, label='Humidity (%)')
            else:
                ax5.scatter(data['Ambient_Temperature_C'], 
                           data['Milk_Yield_L'],
                           alpha=0.6, 
                           s=20, 
                           color='steelblue')
            
            ax5.set_xlabel('Ambient Temperature (°C)')
            ax5.set_ylabel('Milk Yield (L/day)')
            ax5.set_title('Temperature Impact on Milk Yield', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Environmental Data Not Available', 
                    ha='center', va='center')
            ax5.set_title('Temperature Impact on Milk Yield')
        
        # ====================================================================
        # PLOT 6: Feed & Water Intake Analysis (Bottom Left)
        # ====================================================================
        ax6 = fig.add_subplot(gs[2, 0])
        
        if 'Feed_Quantity_kg' in data.columns and 'Water_Intake_L' in data.columns:
            # Scatter plot
            if has_target:
                scatter = ax6.scatter(data['Feed_Quantity_kg'], 
                                     data['Water_Intake_L'],
                                     c=data['Milk_Yield_L'], 
                                     cmap='RdYlGn', 
                                     alpha=0.6, 
                                     s=20)
                plt.colorbar(scatter, ax=ax6, label='Milk Yield (L)')
            else:
                ax6.scatter(data['Feed_Quantity_kg'], 
                           data['Water_Intake_L'],
                           alpha=0.6, 
                           s=20, 
                           color='steelblue')
            
            ax6.set_xlabel('Feed Quantity (kg/day)')
            ax6.set_ylabel('Water Intake (L/day)')
            ax6.set_title('Feed vs Water Intake', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Feed/Water Data Not Available', 
                    ha='center', va='center')
            ax6.set_title('Feed vs Water Intake')
        
        # ====================================================================
        # PLOT 7: Activity & Behavior Patterns (Bottom Center)
        # ====================================================================
        ax7 = fig.add_subplot(gs[2, 1])
        
        activity_features = {
            'Walking_Distance_km': 'Walking',
            'Grazing_Duration_hrs': 'Grazing',
            'Rumination_Time_hrs': 'Rumination',
            'Resting_Hours': 'Resting'
        }
        
        available_activity = {k: v for k, v in activity_features.items() if k in data.columns}
        
        if available_activity:
            activity_means = [data[col].mean() for col in available_activity.keys()]
            activity_labels = list(available_activity.values())
            
            colors_activity = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            ax7.bar(activity_labels, activity_means, color=colors_activity[:len(activity_labels)])
            ax7.set_ylabel('Average Hours/Distance')
            ax7.set_title('Average Activity & Behavior Patterns', fontweight='bold')
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3, axis='y')
        else:
            ax7.text(0.5, 0.5, 'Activity Data Not Available', 
                    ha='center', va='center')
            ax7.set_title('Activity & Behavior Patterns')
        
        # ====================================================================
        # PLOT 8: Days in Milk vs Yield (Bottom Right)
        # ====================================================================
        ax8 = fig.add_subplot(gs[2, 2])
        
        if has_target and 'Days_in_Milk' in data.columns:
            # Scatter plot with trend line
            ax8.scatter(data['Days_in_Milk'], 
                       data['Milk_Yield_L'],
                       alpha=0.5, 
                       s=20, 
                       color='steelblue')
            
            # Add trend line
            z = np.polyfit(data['Days_in_Milk'].dropna(), 
                          data.loc[data['Days_in_Milk'].notna(), 'Milk_Yield_L'], 
                          2)
            p = np.poly1d(z)
            x_trend = np.linspace(data['Days_in_Milk'].min(), 
                                 data['Days_in_Milk'].max(), 100)
            ax8.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
            
            ax8.set_xlabel('Days in Milk (Lactation Stage)')
            ax8.set_ylabel('Milk Yield (L/day)')
            ax8.set_title('Lactation Curve (Days in Milk vs Yield)', fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Lactation Data Not Available', 
                    ha='center', va='center')
            ax8.set_title('Lactation Curve')
        
        # Overall title
        fig.suptitle('Cattle Milk Yield Prediction - Comprehensive Data Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save figure
        plt.savefig('summary_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Comprehensive cattle milk yield visualization created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_fallback_plot(data: pd.DataFrame) -> bool:
    """
    Create a simple fallback plot when comprehensive visualization fails
    """
    try:
        setup_plot_style()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Try to plot milk yield if available
        if 'Milk_Yield_L' in data.columns:
            sns.histplot(data['Milk_Yield_L'], kde=True, ax=ax, color='steelblue')
            ax.set_title('Milk Yield Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Milk Yield (Liters/Day)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        else:
            # Use any available numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sns.histplot(data[numeric_cols[0]], kde=True, ax=ax)
                ax.set_title(f'Distribution of {numeric_cols[0]}')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel('Frequency')
            else:
                ax.text(0.5, 0.5, 'No numeric data available for visualization', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('No Data Available')
        
        plt.tight_layout()
        plt.savefig('summary_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Fallback visualization created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating fallback visualization: {str(e)}")
        return False

def main():
    """
    Main visualization function for cattle milk yield prediction pipeline
    """
    # Get input file from command line argument
    if len(sys.argv) < 2:
        print("Error: Please provide input CSV file path as argument")
        print("Usage: python visualize.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Load the preprocessed data
        print(f"Loading data for visualization: {input_file}")
        data = pd.read_csv(input_file)
        print(f"Data shape: {data.shape}")
        
        if data.empty:
            print("❌ Empty dataset - cannot create visualization")
            sys.exit(1)
        
        # Display dataset info
        print(f"Columns: {len(data.columns)}")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        print(f"Numeric columns: {len(numeric_cols)}")
        
        # Attempt to create comprehensive cattle-specific visualization
        success = create_cattle_milk_yield_plot(data)
        
        # Fallback if comprehensive plot fails
        if not success:
            print("Attempting fallback visualization...")
            success = create_fallback_plot(data)
        
        if success:
            print("\n✅ Visualization saved as: summary_plot.png")
            print(" Generated comprehensive cattle milk yield analysis dashboard")
            print("   - 8 panels covering: yield distribution, vaccinations, correlations,")
            print("     physical characteristics, environmental factors, feed/water,")
            print("     activity patterns, and lactation curves")
            
            # Call next script in pipeline
            print(f"\n🔗 Calling next script: cluster.py")
            import subprocess
            subprocess.run(["python", "cluster.py", input_file])
        else:
            print("❌ Failed to create visualization")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()