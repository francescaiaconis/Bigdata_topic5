import subprocess
import sys
import os

def run_python_script(script_name):
    print(f"Running Python script: {script_name}")
    result = subprocess.run(['python', script_name], capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    else:
        print(f"Successfully ran {script_name}")

def run_spark_script(script_name):
    print(f"Running Spark script: {script_name}")
    spark_submit_path = r'C:\spark-3.4.3-bin-hadoop3\bin\spark-submit'  # Percorso completo a spark-submit
    spark_submit_command = [spark_submit_path, '--master', 'local', '--deploy-mode', 'client', script_name]
    print(f"Running command: {' '.join(spark_submit_command)}")
    result = subprocess.run(spark_submit_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name} with spark-submit:\n{result.stderr}")
        sys.exit(1)
    else:
        print(f"Successfully ran {script_name} with spark-submit")

if __name__ == "__main__":
    scripts = [
        ('python', 'pre_profiling.py'),
        ('python', 'missing_values.py'),
        ('spark', "preprocessing.py"),
        ('python', 'post_profiling.py'),
        ('python', 'classification.py')
    ]
    
    for script_type, script in scripts:
        if not os.path.isfile(script):
            print(f"File {script} does not exist.")
            sys.exit(1)
            
        if script_type == 'python':
            run_python_script(script)
        elif script_type == 'spark':
            run_spark_script(script)
        else:
            print(f"Unknown script type: {script_type}")
            sys.exit(1)
