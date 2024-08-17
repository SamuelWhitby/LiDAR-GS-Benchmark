<div align="center">
    <h1>LiDAR-GS: An Industry Derived Ground Segmentation Benchmark</h1>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8.10-EDEDED?logo=python&logoColor=ffdd54&labelColor=3670A0" /></a>
    <a href="https://ubuntu.com/"><img src="https://img.shields.io/badge/Ubuntu-20.04-EDEDED?logo=Ubuntu&logoColor=white&labelColor=E95420" /></a>
    <a href="https://www.open3d.org/"><img src="https://img.shields.io/badge/Open3D-0.18.0-2D2D2D?logo=https://raw.githubusercontent.com/SamuelWhitby/LiDAR-GS-Benchmark/main/images/open3d-icon.svg&logoColor=ffffff&labelColor=2D2D2D&color=EDEDED" alt="Open3D Version" /></a>

  <br />
  <br />   
  <p align="center"><img src=images/ground-labelling.png alt="Ground Segmentation" width="950px"></p>
  
  Welcome to the **LiDAR** **G**round **S**egmentation (**LiDAR-GS**) Benchmark, an industry-derived evaluation tool to holistically assess the performance of ground segmentation algorithms across a wide range of scenarios and sensors. 
</div>

## Updates
*  14/08/2024 v1.0 release

## Contents
0. [Aims & Objectives](#Aims-&-Objectives)
2. [What's Inside](#What's-Inside)
3. [Prerequisite Packages](#Prerequisite-Packages)
4. [How to Run](#How-to-Run)
5. [Evaluation](#Evaluation)
6. [User Experience](#User-Experience)
7. [References](#References)

## Aims & Objectives
The aim of this benchmark was to create a ground segmentation evaluation tool which reflects the desires of the industry and its stakeholders. Utilising this package, developers can build algorithms which are robust to a diverse range of scenearios and sensors, ensuring that they are future-proof, safe and non-discriminatory. The objectives were:

### Evaluation 
* To evaluate ground segmentation algorithms against scenarios and metrics reflective of industry needs.
* To generate a performance score for urban and rural scenarios and sensor type and position, as well as provide an overall score.
* To create a leaderboard to compare algorithm results against.

### User Experience
* To provide an intuitive algorithm integration experience with clear instructions.
* To create an inclusive environment for users with different needs.
* To create a visually pleasing user interface, which is easy to navigate and understand.

## What's Inside
The benchmark consists of several datasets and evaluates four pertinent ground segmentation algorithms. This is what you can find inside!
### Folder Structure
<p align="left"><img src=images/folder-structure.svg alt="Folder Structure" width="800px" /></p>

### Datasets
*  [SemanticKITTI](https://www.semantic-kitti.org/) - Represents Urban Scenarios.
*  [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D?tab=readme-ov-file) - Represents Rural Scenarios.
* [LiDAR-CS](https://github.com/LiDAR-Perception/LiDAR-CS) - Provides a variety of Sensor Heights and Types.

### Algorithms Evaluated
*  [GndNet](https://github.com/anshulpaigwar/GndNet) - Machine Learning-Based Algorithm
*  [Ground Plane Fitting](https://github.com/JonasHablitzel/PyGroundSegmentation) (GPF) - Ground Modelling-Based Algorithm
*  [LineFit](https://github.com/Kin-Zhang/linefit) - Ground Modelling-Based Algorithm
*  [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) - Ground Modelling-Based Algorithm

## Prerequisite Packages
> To utilise the package, very few dependencies are required.
```commandline
sudo apt-get install python3-pip python3-dev -y
```
### Python
> The following libraries are required to run the evaluate and visualise scripts.
```commandline
pip install numpy pandas openpyxl psutil gputil open3d
```
### Example Evaluations
**NOTE:** If you wish to run any of the example scripts, please follow the links provided in the **Algorithms Evaluated** section and follow their setup instructions.
## How to Run
### Test Environment
The code was tested successfully on:
* Ubuntu 20.04
* Open3D 0.18.0
* Python 3.8.10
### Setup & Run
Prior to running the script, users are required to input their algorithm into the following "evaluate_template.py" lines:
```commandline
13.  # Insert Algorithm Libraries
18.  # Assign Algorithm Name - This must be consistent with the results.xlsx algorithm name.
107. # Run Initiate Algorithm Function
172. # Run Ground Estimation Function
```
> Once complete, enter the package directory and run the "evaluate_template.py" script:
```commandline
cd /path/to/your/LiDAR-CS
python3 evaluate_template.py
```
## Evaluation
The ground segmentation evaluation can be viewed through two mediums: visually or numerically. For instances where the algorithm performance falls below the adjustable IoU threshold of 70%, users can visualise the point cloud as demonstrated below. This allows developers to identify areas for improvement in their algorithm approach, highlighting points of over and under-segmentation.

### Visualise
The terminal output and visualised point cloud of the LineFit algorithm can be seen below. The colours provide a clear contrast in under and over-segmented areas, as well as highlighting where the algorithm successfully identified the ground plane. 
<br />   
<p align="center"><img src=images/linefit-visualised.png alt="Visualise" width="950px" /></p>

### Analyse
Alternatively users can analyse the algorithmic performance numerically through the "results.xlsx" file. This file gathers results for 13 individual attributes against 8 performance metrics. Each attribute is made up of 10 samples of 32 sequential scans (with the exception of the sensor data, which is 10 samples of 1 scan due to requiring further labelling.) The attributes and metrics assessed can be seen below:
#### Attributes
Prior to sampling, the distribution of each attribute scans across the three datasets was:
<br />   
<p align="left"><img src=images/attribute-population.png alt="Visualise" width="800px" /></p>

#### Metrics
*  Intersect over Union (IoU)     [%]
*  Precision (PRE)                [%]
*  Recall (REC)                   [%]
*  Coefficient of Variation (CV)  [%]
*  CPU Usage                      [%]
*  GPU Usage                      [%]
*  Memory Usage                   [%]
*  Speed                          [Hz]

## User Experience
User experience was a top priority when creating the benchmark. This included making sure the product was inclusive and adaptable to people's needs, that all instructions were clear, simple and intuitive, and that the outputs were aesthetically pleasing. To address the user experience, the following features were added to improve usability:
*  +/- to adjust the datapoint size.
*  n/b to quickly navigate the scans.
*  Loading bar and tasks complete indicator.
*  Optimally positioned camera to view the point cloud segmentation.
*  Option to add additional datasets without altering the evaluation script.

The final feature added to the benchmark was the option to alter the colour palette of the visualise code to the IBM Design library colour blind palette (left), greyscale palette (right) or the option to customise the palette to your needs. The output of these alternative colours can be seen below:
<br />   
<p align="center"><img src=images/colour-palettes.png alt="Colourblind Colour Palette" width="950px" /></p>

## References
I would like to thank the following for making their work open source, and I wish to continue this trend so that others can gain insight into current ground segmentation approaches.
```
@inproceedings{himmelsbach2010fast,
  title={Fast segmentation of 3d point clouds for ground vehicles},
  author={Himmelsbach, Michael and Hundelshausen, Felix V and Wuensche, H-J},
  booktitle={Intelligent Vehicles Symposium (IV), 2010 IEEE},
  pages={560--565},
  year={2010},
  organization={IEEE}
}
```
```
@inproceedings{lee2022patchworkpp,
    title={{Patchwork++: Fast and robust ground segmentation solving partial under-segmentation using 3D point cloud}},
    author={Lee, Seungjae and Lim, Hyungtae and Myung, Hyun},
    booktitle={Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.},
    year={2022},
    note={{Submitted}} 
}
```
```
@inproceedings{paigwar2020gndnet,
  title={GndNet: Fast Ground Plane Estimation and Point Cloud Segmentation for Autonomous Vehicles},
  author={Paigwar, Anshul and Erkent, {\"O}zg{\"u}r and Gonz{\'a}lez, David Sierra and Laugier, Christian},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```
```
@misc{jiang2020rellis3d,
      title={RELLIS-3D Dataset: Data, Benchmarks and Analysis}, 
      author={Peng Jiang and Philip Osteen and Maggie Wigness and Srikanth Saripalli},
      year={2020},
      eprint={2011.12954},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@article{fang2023lidar,
  title={LiDAR-CS Dataset: LiDAR Point Cloud Dataset with Cross-Sensors for 3D Object Detection},
  author={Fang, Jin and Zhou, Dingfu and Zhao, Jingjing and Tang, Chulin and Xu, Cheng-Zhong and Zhang, Liangjun},
  journal={arXiv preprint arXiv:2301.12515},
  year={2023}
}
```
```
@inproceedings{behley2019iccv,
  author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
  year = {2019}
}
```
```
@inproceedings{geiger2012cvpr,
  author = {A. Geiger and P. Lenz and R. Urtasun},
  title = {{Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite}},
  booktitle = {Proc.~of the IEEE Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3354--3361},
  year = {2012}
}
```

## Future Work
*  To label the ground plane for all LiDAR-CS dataset scans.
*  To populate the leaderboard with a variety of algorithm approaches.
*  To add additional spoken languages.
*  To add additional programming languages.
