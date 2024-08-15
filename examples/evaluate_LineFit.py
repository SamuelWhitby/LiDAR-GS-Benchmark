import os
import sys
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import time
import psutil
import GPUtil
import visualise
### -- Algorithm Specific Libraries -- ### <----------------------------------------------------------------------------------------

from linefit import ground_seg

### -- Adjustable Variables -- ###

## Algorithm Name ##
algorithm_name = 'LineFit' # <------------------------------------------------------------------------------------------------------
IoU_alert_threshold = 0.7


### -- Setup -- ###

## Directories ##
cur_dir = os.path.dirname(os.path.abspath(__file__))
lidar_gs_dir = os.path.abspath(os.path.join(cur_dir, '..'))
base_dir = os.path.join(lidar_gs_dir, 'scans/')
ontology_file = os.path.join(lidar_gs_dir, 'ontology.csv')
results_file = os.path.join(cur_dir, 'results.xlsx')
visulalisation = os.path.join(lidar_gs_dir, 'visualise.py')

## Ontology ##
ontology_df = pd.read_csv(ontology_file)
category_col = 'category'
output_value_col = 'output_value'
ground_labels = ontology_df[ontology_df['category'].str.contains('ground', case=False, na=False)][output_value_col].tolist()


### -- Functions -- ###

## Read Bin File ##
def read_bin(data_dir, file):
    bin_path = os.path.join(data_dir, file)
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

## Read Label File ##
def read_label(label_dir, base_name):
    label_path = os.path.join(label_dir, base_name + '.label')
    label_file = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    return label_file

## Get System Usage ##
def getSystemUsage():
    # Capture CPU, memory, and GPU usage
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    gpu_info = GPUtil.getGPUs()
    gpu_usage = [gpu.load * 100 for gpu in gpu_info]
    
    return cpu_usage, memory_usage, gpu_usage

## Get Attribute Name ##
def getAttributeName(base_dir):
    path_components = base_dir.split(os.sep)  # split path into components
    
    attribute_index = path_components.index('scans') + 3  # +2 to skip 'scene_attributes' and the next folder level
    attribute_name = path_components[attribute_index]
    attribute_name = attribute_name.replace('_', ' ').title()
    return attribute_name

## Find Next Available Sheet Name ##
def getNextSheetName(writer, base_name):
    sheet_names = writer.book.sheetnames
    suffix = 1
    
    # Check for the base name first without suffix
    if base_name not in sheet_names:
        return base_name
    
    # If base name exists, find the next available suffix
    while f"{base_name} {suffix}" in sheet_names:
        suffix += 1
        
    return f"{base_name} {suffix}"

## Loading Bar ##
def getLoadingBar(task_counter, total_tasks):
    bar_length = 100  # Length of the loading bar
    completed_length = int(bar_length * task_counter / total_tasks)
    bar = '#' * completed_length + '-' * (bar_length - completed_length)
    sys.stdout.write(f"\r[{bar}] {task_counter}/{total_tasks} tasks completed")
    sys.stdout.flush()


### -- Initial Message -- ###

print('\n' + '-' * 100)
print('\nThank you for using the LiDAR-GS benchmark.\nPlease ensure the results.xlsx file is closed whilst your algorithm is evaluated.')
print('\n' + '-' * 100 + '\n')


### -- Intiating Algorithm -- ###

## Initiate Ground Segmentation Algorithm ## <-------------------------------------------------------------------------------
config_path = os.path.join(cur_dir, 'config.toml')
groundseg = ground_seg(config_path)

### -- Evaluate Algorithm Across All Attributes -- ###

all_results = []
low_accuracy_folders = []

# Count total number of folders for the loading bar
environment_path = os.path.join(base_dir, 'scene_attributes/urban/')  # LineFit requires heights to be set manually

total_tasks = 0

for attribute_folder in os.listdir(environment_path):
    attribute_path = os.path.join(environment_path, attribute_folder)

    # Count the number of group folders for the loading bar
    total_tasks += len(os.listdir(attribute_path))

task_counter = 0

for attribute_folder in os.listdir(environment_path):
    attribute_path = os.path.join(environment_path, attribute_folder)
    attribute_name = getAttributeName(attribute_path)

    for group_folder in sorted(os.listdir(attribute_path)):
        data_dir = os.path.join(attribute_path, group_folder, 'bin/')
        label_dir = os.path.join(attribute_path, group_folder, 'labels/')
        frames = os.listdir(data_dir)

        ## Loading Bar ##
        getLoadingBar(task_counter, total_tasks)

        ## Metric Accumulators ##
        metrics = {
            'cpu_usage': 0,
            'gpu_usage': 0,
            'memory_usage': 0,
            'IoU_values': [],
            'PRE_values': [],
            'REC_values': []
        }

        ### -- Evaluating Algorithm Against Attribute -- ###

        for f in range(len(frames)):
            base_name = "%06d" % f

            ## Get Point Cloud Data ##
            pcd_path = os.path.join(data_dir, "%06d.bin" % f)
            pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
            
            ### -- Get System Paramters - Efficiency & Response -- ###
            system_usage_before = getSystemUsage()
            start_time = time.time()
            ground_estimate = np.array(groundseg.run(pcd[:, :3]))  # run ground estimation algorithm <-------------------------------------------------------------------------------
            end_time = time.time()
            system_usage_after = getSystemUsage()

            ### -- Get System Paramters - Efficiency & Response -- ###
            speed           = 1/(end_time - start_time)
            cpu_usage       = system_usage_after[0] - system_usage_before[0]
            gpu_usage       = system_usage_after[1] - system_usage_before[1]
            memory_usage    = [after - before for before, after in zip(system_usage_before[2], system_usage_after[2])] if system_usage_before and system_usage_after else [0]

            ## Accumulate System Usage ##
            metrics['cpu_usage'] += cpu_usage
            metrics['gpu_usage'] += gpu_usage
            if memory_usage:
                metrics['memory_usage'] += memory_usage[0]
            else:
                metrics['memory_usage'] += 0

            ### -- Get Performance Parameters - Accuracy -- ###

            ## Get Ground Estimate and Ground Truth ##
            ground_truth = np.isin(read_label(label_dir, base_name), ground_labels)
            
            if base_name == "000000":  # get the ground at 000000 for visualisation
                gnd_tru = ground_truth
                gnd_est = ground_estimate

            ground_estimate, ground_truth = 1-ground_estimate, 1-ground_truth  # invert ground points

            intersection = np.logical_and(ground_truth, ground_estimate)
            union = np.logical_or(ground_truth, ground_estimate)

            IoU = np.sum(intersection) / np.sum(union)
            precision = np.sum(intersection)/ground_estimate.sum()
            recall = np.sum(intersection)/ground_truth.sum()

            ## Calculate IoU and store it ##
            metrics['IoU_values'].append(IoU)
            metrics['PRE_values'].append(precision)
            metrics['REC_values'].append(recall)

        ## Track Low IoU ##
        if IoU < IoU_alert_threshold:
            ## Append Identifiers ##
            low_accuracy_folders.append({
                'Attribute': attribute_name,
                'Group': group_folder,
                'IoU': IoU,
                'Directory': os.path.join(attribute_path, group_folder),
                'Ground Truth': gnd_tru,
                'Ground Estimate': gnd_est
                })

        ## Update and Print Loading Bar ##
        task_counter += 1

    ### -- Calculate Averages -- ###

    ## System Parameters ##
    cpu_usage = metrics['cpu_usage'] / len(frames) if len(frames) > 0 else 0
    gpu_usage = metrics['gpu_usage'] / len(frames) if len(frames) > 0 else 0
    memory_usage = metrics['memory_usage'] / len(frames) if len(frames) > 0 else 0

    ## Accuracy Parameters ##
    precision = np.mean(metrics['PRE_values']) if metrics['PRE_values'] else 0
    recall = np.mean(metrics['REC_values']) if metrics['REC_values'] else 0
    IoU = np.mean(metrics['IoU_values']) if metrics['IoU_values'] else 0
    std_IoU = np.std(metrics['IoU_values']) if metrics['IoU_values'] else 0
    CVariation = (std_IoU / IoU) * 100 if IoU > 0 else 0

    ### -- Store Results -- ###

    results = ({
        'Attribute': getAttributeName(attribute_path),
        'IoU [%]': IoU*100,                
        'PRE [%]': precision*100,
        'REC [%]': recall*100,
        'CV [%]': CVariation,
        'CPU [%]': cpu_usage,
        'GPU [%]': gpu_usage,
        'MEM [%]': memory_usage,
        'Speed [Hz]': speed,
    })

    all_results.append(results)


### -- Export Results -- ###
print('\n\n' + '-' * 100)

## Convert Results to DataFrame ##
results_df = pd.DataFrame(all_results)

try:
    ## Append to or create the Excel file ##
    if os.path.exists(results_file):
        with pd.ExcelWriter(results_file, engine='openpyxl', mode='a') as writer:
            new_sheet_name = getNextSheetName(writer, algorithm_name)
            results_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    else:
        with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
            new_sheet_name = f"{algorithm_name} Evaluation 1"
            results_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    
    print('\nPlease see the results.xlsx file for algorithm results')

    ## Export Low Accuracy Scenarios to CSV ##
    if low_accuracy_folders:
        print('\nThe low accuracy attributes were:')
        for entry in low_accuracy_folders:
            print(f"Attribute: {entry['Attribute']}, Group: {entry['Group']}, IoU: {entry['IoU']:.2f}")
        
        print('\nWould you like to view the low accuracy attribute scenarios?')
        print('\nPress...')
        print('\n\t [Y] Yes')
        print('\n\t [N] No')
        print('\n' + '-' * 100)
        user_input = input().strip().upper()

        if user_input == 'Y':
            print("\nStarting visualisation...")
            visualise.visualise_low_accuracy_scenarios(low_accuracy_folders)
            
        elif user_input == 'N':
            print("\nVisualisation skipped.")
        else:
            print("\nInvalid input. Visualisation skipped.")
    else:
        print("\nThere were zero instances of accuracy falling below the threshold")

except Exception as e:
    print(f"\nAn error occurred while exporting the results: {e}")
    print('\n' + '-' * 100)