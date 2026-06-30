import subprocess
import os

def run_script(script_name):
    print(f"Running {script_name}...")
    res = subprocess.run(['.venv/Scripts/python.exe', script_name], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  {script_name} completed successfully.")
        print(res.stdout)
    else:
        print(f"  {script_name} FAILED with code {res.returncode}:")
        print(res.stderr)
    return res.returncode == 0

def verify_all():
    scripts = [
        'feature_sdist_eval.py',
        'bifurcation_fitting_example.py',
        'plot_pred_example.py',
        'plot_figure2.py',
        'plot_figure3.py'
    ]
    
    all_success = True
    for s in scripts:
        success = run_script(s)
        if not success:
            all_success = False
            
    # Check generated files
    expected_outputs = [
        'Core Algorithms and Examples/Feature and Sleep distance computation/SleepDistance_Python_Plot.png',
        'Core Algorithms and Examples/Examples Bif Fitting/Bifurcation_Python_Plot.png',
        'Core Algorithms and Examples/Example Prediction Tipping point/Prediction_Python_Plot.png',
        'Final Plot codes/Figure2/Figure2a_Python.png',
        'Final Plot codes/Figure3/Figure3ad_Python.png'
    ]
    
    print("\nVerifying Output Files:")
    for out in expected_outputs:
        if os.path.exists(out):
            print(f"  [FOUND] {out}")
        else:
            print(f"  [MISSING] {out}")
            all_success = False
            
    if all_success:
        print("\nAll Python conversion scripts executed successfully and generated all figures!")
    else:
        print("\nVerification failed. Some files were missing or scripts failed to run.")

if __name__ == "__main__":
    verify_all()
