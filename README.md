# ALMA-Disk-Code
A collection of python utilities for analyzing ALMA data



Here are a collection of utilities, and notes, that I have put together for analyzing ALMA data, primarily of primordial/debris disks. These include:


quick_disk.py: A python script for doing a quick fit to a disk image. Using concentric rings of emission, this code fits to either the continuum (or moment 0) map or the full line profile to derive the radial emission profile. Useful for an inital quick look at the structure of a disk (e.g. does it have an inner hole? Is the emission one wide ring or multiple narrow rings?)


var_vis.py: A python script for deriving the weights of visibility data based on the dispersion of the visibilities. 



alma_chisq.pdf: A introduction to manipulating ALMA visibility data within IDL or Python. Focused mainly on how to read the data into your favorite programming environment.


cluster_guide.pdf: Not strictly related to ALMA, but useful nonetheless... For those at Wesleyan University, this is a guide to running code on the local computing cluster, including how to set up serial/parallel jobs, and useful functions to monitor the progress of your code.