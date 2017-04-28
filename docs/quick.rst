.. _quick:

Quick Start Guide
=================

Install Python
----------------
The LMR codebase utilizes many packages typical to the scientific Python stack.
It is recommended that a Python distribution and package manager is installed
to fulfill program dependencies.  Install the latest 2.7.x version of Python.

* `Anaconda <https://www.continuum.io/downloads>`_ (recommended) provides
  many of the required packages and many more useful packages pre-installed.
* `Miniconda <http://conda.pydata.org/miniconda.html>`_ is a barebones
  installation where you can then add only the necessary packages.

It will be necessary to add a few extra packages. E.g. ::

    conda install netCDF4
    conda install -c https://conda.anaconda.org/ajdawson pyspharm

If at any point there is an error for a missing dependency, install it using
``conda install`` command (the installer that comes with Anaconda).


Retrieve LMR source code
------------------------
Navigate to the directory where you want to store the LMR code and retrieve
it from the repostory.

Using Github::

    git clone git@github.com:frodre/LMR.git

Using SVN::

    svn co https://www.atmos.washington.edu/svn/lmr/tags/v2.0

Retrieve Data
-------------
Before running an experiment, the source data must be downloaded and referenced.

First you must have the data in position to perform an experiment. Download this
tar file: `LMR_data_control.tar <http://www.atmos.washington.edu/~hakim/lmr_data/LMR_data_control.tar>`_
and move it to a directory where you will unpack it; here we will call that
directory /home/disk/foo/LMR. This directory must be readable from wherever you
plan to perform the experiment. ::

    tar -xvf LMR_data_control.tar

will give you something that looks like this in the /home/disk/foo/LMR
directory ::

    data/  LMR_data_control.tar  PSM/

You need to softlink one file in the data/model/ccsm4_last_millenium/
subdirectory ::

    cd data/model/ccsm4_last_millenium/
    ln -s tas_Amon_CCSM4_past1000_085001-185012.nc tas_sfc_Amon_CCSM4_past1000_085001-185012.nc

Running an Experiment
---------------------

To run an experiment, you must edit configuration. In the LMR code directory
copy ``config_template.yml``, which is under version control, to ``config .yml``,
which is not. ::

    cp config_template.yml config.yml

Now edit ``config.yml``. At a minimum, do the following:

.. The existence requirement below should be verified [THIS IS A COMMENT]

1. set ``core.lmr_path`` to the directory containing the proxy, prior, and
   calibration data for the experiment (see details below). If you are at UW,
   use R. Tardif's path as set by default in the config file. Otherwise, use the
   /home/disk/foo/LMR path where you unpacked the tar file in the steps described
   above.

2. set ``core.datadir_output`` to the working directory output for LMR. This is where
   data will be written during the experiment, so local disk is better (faster)
   than over a network. If the directory does not exist, you need to create it
   before running an experiment.


3. set ``core.archive_dir`` to the LMR reconstruction archive directory. This is where
   the experiment is archived, and datadir_output is scrubbed clean
   (frees up local disk space; often the archive is located across a network,
   but speed is no longer an issue). Note that data is lost in this step
   (ensemble mean from full ensemble), which you may wish to change at some
   point. Again, if the directory does not exist, you need to create it before running an
   experiment.

You are now ready to run an experiment from the code directory! ::

    python LMR_wrapper.py


..  note::  More steps are involved to use the NCDC proxy database,
    pre-calculated obs estimate files, and certain psm pre-calibration files
    that will speed up the reconstruction process. [PLACE LINKS TO
    DOCUMENTATION PAGES IN THIS NOTE]
