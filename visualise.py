import os
import numpy as np
import pandas as pd
import open3d as o3d


### -- Setup -- ###

## Visualisation Colors ##
TP_colour = [1, 1, 0]  # yellow
TN_colour = [1, 1, 1]  # white
FP_colour = [0, 1, 0]  # green
FN_colour = [1, 0, 0]  # red

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

def assign_colours(directory, ground_truth, ground_estimate):
    
    ## Process the Selected File ##
    file_name='000000.bin'
    data_dir = os.path.join(directory, 'bin')
    pcd = read_bin(data_dir, file_name)  # Load point cloud data
    
    ## Visualisation ##
    colors = np.zeros((pcd.shape[0], 3))
    colors[(ground_truth & ground_estimate)] = TP_colour    # TP: True Positive (Yellow)
    colors[(~ground_truth & ~ground_estimate)] = TN_colour  # TN: True Negative (White)
    colors[(~ground_truth & ground_estimate)] = FP_colour   # FP: False Positive (Green)
    colors[(ground_truth & ~ground_estimate)] = FN_colour   # FN: False Negative (Red)
    
    ## Create the Point Cloud ##
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    
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

    print("\nColor Key:")
    print("\n\tYellow: TP (True Positive)")
    print("\n\tWhite: TN (True Negative)")
    print("\n\tGreen: FP (False Positive)")
    print("\n\tRed: FN (False Negative)")
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

        print(f"\nVisualizing Attribute: {attribute}, Group: {group}, IoU: {IoU*100:.2f}%")
        print('\n' + '-' * 100)

        pcd_o3d = assign_colours(directory, ground_truth, ground_estimate)
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
    visualise_low_accuracy_scenarios([
    ])