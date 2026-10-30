# Third-Party Models And Attribution

## MuJoCo Menagerie

Robot models are from Google DeepMind's public
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
The original model XML and assets are retained in `vendor/mujoco_menagerie`.
The KUKA iiwa 14 model carries its BSD-3-Clause license; the Sharpa Wave model
carries its Apache-2.0 license. Their original model-specific LICENSE files and
READMEs are included in the runnable source package. Assembly and workcell
changes are made by this project's scene builders, not by modifying the vendor
model files.

## YCB Object And Model Set

The public processed meshes are from the
[official YCB download index](https://ycb-benchmarks.s3.amazonaws.com/index.html).
The index identifies the dataset license as
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Included objects are `003_cracker_box`, `005_tomato_soup_can`, and
`010_potted_meat_can`. The simulation uses the processed nontextured meshes,
recolors them for its declared appearance prior, and assigns simulated mass and
contact parameters. These parameters and appearances are not physical sensor
or measured-object specifications.

Dataset credit: Berk C. Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa,
Pieter Abbeel, and Aaron M. Dollar. See the official index for the related 2015
ICAR and IEEE Robotics & Automation Magazine publications. The redistribution
preserves this attribution and the source/license links.
