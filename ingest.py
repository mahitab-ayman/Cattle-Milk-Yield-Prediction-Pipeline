import sys
import pandas as pd
import os

# FORCE OUTPUT BUFFERING
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

def main():
    """
    Main ingestion function for cattle milk yield prediction dataset
    Loads data and passes it to preprocessing pipeline
    """
    print("="*70, flush=True)
    print("CATTLE MILK YIELD PREDICTION - DATA INGESTION", flush=True)
    print("="*70, flush=True)
    
    # Get file path from command line argument
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide dataset file path", flush=True)
        print("Usage: python ingest.py <path_to_csv_file>", flush=True)
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"Argument received: {file_path}", flush=True)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found: {file_path}", flush=True)
        sys.exit(1)
    
    print(f"\n Loading dataset from: {file_path}", flush=True)
    
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
        print(f"✓ Successfully loaded dataset", flush=True)
        print(f"  - Rows: {data.shape[0]:,}", flush=True)
        print(f"  - Columns: {data.shape[1]}", flush=True)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}", flush=True)
        sys.exit(1)
    
    # Save raw data
    output_file = "data_raw.csv"
    try:
        data.to_csv(output_file, index=False)
        print(f"\n Raw data saved as: {output_file}", flush=True)
        # Verify file was created
        if os.path.exists(output_file):
            print(f"✓ Verified: {output_file} exists", flush=True)
        else:
            print(f"❌ ERROR: {output_file} was not created!", flush=True)
    except Exception as e:
        print(f"❌ Error saving raw data: {e}", flush=True)
        sys.exit(1)
    
    # Call next script in pipeline
    print("\n" + "="*70, flush=True)
    print(" Calling next script: preprocess.py", flush=True)
    print("="*70 + "\n", flush=True)
    
    try:
        import subprocess
        # Use unbuffered output for subprocess
        result = subprocess.run(
            [sys.executable, "-u", "preprocess.py", output_file],
            capture_output=True,
            text=True
        )
        print(f"preprocess.py stdout: {result.stdout}", flush=True)
        print(f"preprocess.py stderr: {result.stderr}", flush=True)
        print(f"preprocess.py return code: {result.returncode}", flush=True)
        
    except Exception as e:
        print(f"❌ Error calling preprocess.py: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    print("=== INGEST.PY COMPLETED ===", flush=True)