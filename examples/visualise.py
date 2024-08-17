import os
import numpy as np
import pandas as pd
import open3d as o3d


### -- Setup -- ###

## Colour Palettes ##
standard_colour_palette = {
    'TP_colour': [1, 1, 0],  # yellow
    'TN_colour': [1, 1, 1],  # white
    'FP_colour': [0, 1, 0],  # green
    'FN_colour': [1, 0, 0]   # red
}

IBM_colour_palette = {
    'TP_colour': [1, 0.69, 0],        # yellow
    'TN_colour': [1, 1, 1],           # white
    'FP_colour': [0.47, 0.36, 0.94],  # blue
    'FN_colour': [0.86, 0.15, 0.5]    # pink
}

greyscale_colour_palette = {
    'TP_colour': [0.4, 0.4, 0.4],  # grey
    'TN_colour': [0, 0, 0],        # black
    'FP_colour': [0.1, 0.1, 0.1],  # dark grey
    'FN_colour': [1, 1, 1]         # white
}

custom_colour_palette = {           # change colours according to needs
    'TP_colour': [0.7, 0.7, 0.7],   # mid tone between FP and FN
    'TN_colour': [1, 1, 1],         # keep white
    'FP_colour': [0.5, 0.5, 0.5],   # should contrast FN
    'FN_colour': [0.3, 0.3, 0.3]    # should contrast FP
}

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

## Visualise Scenario ##
def visualise_scenario(vis, pcd_o3d):
    # Update the visualisation window
    vis.clear_geometries()
    vis.add_geometry(pcd_o3d)
    vis.poll_events()
    vis.update_renderer()

    ## Adjust Camera Position ##
    view_ctl = vis.get_view_control()
    center = np.asarray(pcd_o3d.get_center())
    view_ctl.set_lookat(center)
    view_ctl.set_front([1, 1, 1.5])  # position
    view_ctl.set_up([0, 0, 1])
    view_ctl.set_zoom(.1)  # distance

def assign_colours(directory, ground_truth, ground_estimate, colour_palette):
    
    ## Process the Selected File ##
    file_name='000000.bin'
    data_dir = os.path.join(directory, 'bin')
    pcd = read_bin(data_dir, file_name)  # Load point cloud data
    
    ## Visualisation ##
    colours = np.zeros((pcd.shape[0], 3))
    colours[(ground_truth & ground_estimate)] = colour_palette['TP_colour']    # TP: True Positive (Yellow)
    colours[(~ground_truth & ~ground_estimate)] = colour_palette['TN_colour']  # TN: True Negative (White)
    colours[(~ground_truth & ground_estimate)] = colour_palette['FP_colour']   # FP: False Positive (Green)
    colours[(ground_truth & ~ground_estimate)] = colour_palette['FN_colour']   # FN: False Negative (Red)
    
    ## Create the Point Cloud ##
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(colours)
    
    return pcd_o3d

### -- Visualise -- ###
def visualise_low_accuracy_scenarios(low_accuracy_folders):
    current_idx = 0

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer", width=800, height=600)

    ## Set Render Options ##
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # set background to black
    opt.point_size = 1.0  # set point size to the smallest possible value

    print("\nColour Key:")
    print("\n\tYellow: TP (True Positive)")
    print("\n\tWhite: TN (True Negative)")
    print("\n\tGreen: FP (False Positive)")
    print("\n\tRed: FN (False Negative)")
    print('\nTo navigate the point clouds, press...')
    print('\n\t [N] Next Scan')
    print('\n\t [B] Previous Scan')
    print('\n\t [Q] Exit')
    print('\n' + '-' * 100)
    
    def update_scan():
        row = low_accuracy_folders[current_idx]
        directory = row['Directory']
        ground_truth = row['Ground Truth']
        ground_estimate = row['Ground Estimate']
        attribute = row['Attribute']
        group = row['Group']
        IoU = row['IoU']

        ground_truth = ground_truth.astype(bool)
        ground_estimate = ground_estimate.astype(bool)

        print('\n', end='')
        print(f"\rVisualising Attribute: {attribute}, Group: {group}, IoU: {IoU*100:.2f}%", end='', flush=True)
        print('\n' + '-' * 100)

        pcd_o3d = assign_colours(directory, ground_truth, ground_estimate, greyscale_colour_palette)  # change colour palette 
        visualise_scenario(vis, pcd_o3d)

    def next_scan(vis):
        nonlocal current_idx
        if current_idx < len(low_accuracy_folders) - 1:
            current_idx += 1
            update_scan()

    def previous_scan(vis):
        nonlocal current_idx
        if current_idx > 0:
            current_idx -= 1
            update_scan()

    vis.register_key_callback(ord("N"), next_scan)
    vis.register_key_callback(ord("B"), previous_scan)

    update_scan()
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # The script can still run standalone if needed by providing a mock low_accuracy_folders list
    visualise_low_accuracy_scenarios()