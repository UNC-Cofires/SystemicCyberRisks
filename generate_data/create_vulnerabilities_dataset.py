#!/usr/bin/env python3
"""
Create Vulnerabilities Dataset

This script runs the complete data generation pipeline in the correct sequence:
1. read_nvd_api.py - Collect CVE data from NIST NVD API
2. merge_files.py - Merge annual vulnerability files into a single dataset
3. pull_description.py - Extract and add vulnerability descriptions

The final output will be saved to ../data/vulnerabilities.parquet
"""

import subprocess
import sys
import os
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Run the script using subprocess
        result = subprocess.run(
            [sys.executable, script_name], 
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,  # Show output in real time
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS: {script_name} completed successfully")
            return True
        else:
            print(f"\n❌ ERROR: {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Failed to run {script_name}: {str(e)}")
        return False

def main():
    """Main function to run the complete data generation pipeline"""
    
    print("🚀 Starting Vulnerability Data Generation Pipeline")
    print("This process will create the complete vulnerabilities dataset")
    print("Note: The NVD API step may take several hours to complete")
    
    start_time = time.time()
    
    # Step 1: Run read_nvd_api.py
    success = run_script(
        "read_nvd_api.py", 
        "Collecting CVE data from NIST NVD API (this may take hours)"
    )
    if not success:
        print("\n💥 Pipeline failed at Step 1 (NVD API collection)")
        sys.exit(1)
    
    # Step 2: Run merge_files.py  
    success = run_script(
        "merge_files.py",
        "Merging annual vulnerability files into single dataset"
    )
    if not success:
        print("\n💥 Pipeline failed at Step 2 (file merging)")
        sys.exit(1)
        
    # Step 3: Run pull_description.py
    success = run_script(
        "pull_description.py",
        "Extracting and adding vulnerability descriptions"
    )
    if not success:
        print("\n💥 Pipeline failed at Step 3 (description extraction)")
        sys.exit(1)
    
    # Calculate total runtime
    end_time = time.time()
    runtime = end_time - start_time
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = int(runtime % 60)
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Final dataset saved to: ../data/vulnerabilities.parquet")
    print(f"⏱️  Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"\nYou can now run the modeling scripts:")
    print(f"  python ../modeling/baseline_abel_koshy_07_25.py")

if __name__ == "__main__":
    main() 