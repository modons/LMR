.. _install:

************************
Installation & LMR setup
************************


Get the LMR code
================

The source code for this project can be found on Github at:
`<https://github.com/modons/LMR>`_

You can either clone directly from the public repository::

    $ git clone https://github.com/modons/LMR.git

Or download the
`source tarball <https://github.com/frodre/LMR/tarball/production>`_::

    $ curl -OL https://github.com/modons/LMR/tarball/master

Installing Python 3
===================
After retrieving the LMR code, you’ll need to set up a Python 3.6+ installation
to run it.  The LMR codebase utilizes many packages typical of the scientific
Python community. For ease of use, we recommended using a Python distribution
such as Anaconda/Miniconda to create your working environment.

* `Anaconda <https://www.anaconda.com/download/>`_ (recommended) provides many
  of the required packages and many more useful packages pre-installed.   This
  option requires a sizeable chunk of disk space.
* `Miniconda <https://conda.io/miniconda.html>`_ is a barebones
  installation with no additional packages.

.. note:: Due to package dependencies of our regridding facility, we do not
  currently support LMR code on Windows OS.

Installing required packages
----------------------------

The Anaconda/Miniconda distributions come with a built in package manager
`conda`, that makes it easy to install/update/remove Python packages.
It also allows for the creation of encapsulated
`environments <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
so you can keep package versions specific to projects!

If you're using Anaconda/Conda package management, we strongly recommend using our
provided environment file to setup the required packages. Under the LMR source
directory you’ll find a .yml file `misc/lmr_pyenv.yml`.  In a terminal, change
to the LMR code directory and use the following command to setup your new Python
environment::

    conda env create -f ./misc/lmr_pyenv.yml

This creates the new python environment, separate from your main installation,
based on the packages listed in the .yml file. The new environment (named
lmr_py3 by default) is located under your Anaconda installation directory in
the 'envs' folder.

Using a Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

*In order to use this environment*, you can use the activate command provided by
conda (recommended), or you can manually prepend the environment directory to
your path environment variable.

.. note:: The activate script provided by conda works by default with  bash for
  versions before conda v4.4.  It can be edited to work with other modern
  shells, but may not work with csh or related shells.

To activate the LMR environment (i.e., set it as the current python target) use
the following::

    $ source activate lmr_py3

*Available option for conda >=4.4*::

    $ conda activate lmr_py3

After this you should see small indicator in your terminal with the name of the
environment, e.g.,::

    (lmr_py3) [user@machine] $

This indicates that you are using the LMR python environment (lmr_py3) in the
current session and that any scripts you run in this session use that
environment by default.

To return/exit to your default Python environment use::

    $ source deactivate

You should see the environment prefix disappear, indicating you're back in your
original Python environment.

.. warning:: if an executable (like ipython) is not in the LMR python
  environment, but your path can still find the executable in your original
  installation you might not notice you're using an older/different version.

.. sample_data::

Retrieving LMR data
===================
Before running an experiment, you’ll have to download some of the source data
for proxies, models, and instrumental analyses.

.. todo: Fix this tar file location for users

Download this tar file:
`LMR_data_control.tar <http://www.atmos.washington.edu/~hakim/lmr_data/LMR_data_control.tar>`_
and move it to a directory where you will unpack it; here we will call that
directory /home/disk/foo/LMR_data. This directory must be readable from wherever
you plan to perform the experiment. Extract files using::

    $ tar -xvf LMR_data_control.tar

giving you something that looks like this in the /home/disk/foo/LMR_data
directory ::

    data/  LMR_data_control.tar  PSM/

This is the default directory structure, which allows the LMR framework to easily look
for data sources in known locations.  However, non-standard data directories
can still be specified in the :ref:`configuration`.

Default folder description
--------------------------

The bulk of the required data exists under the ``data/`` directory ::

    data/
        |-> analyses/
            |-> analysis_exp_folder
                |- analysis_field.nc
                |- ....
        |-> model/
            |-> model_exp_folder
                |- model_field.nc
                |- ....
        |-> proxies/
            |- proxy_db_file.pckl
            |- ....

The analyses folder holds observational analysis experiments used for
calibrating of LMR’s statistical proxy system models (PSMs).  (E.g., NOAA MLOST,
NASA GISTEMP, 20th Century Reanalysis, etc.).  The model folder is where climate
model simulations used for creating a prior are stored. (E.g., various CMIP5
simulations). And finally, the proxies folder is where the proxy databases
(pandas dataframes created using LMR_proxy_preprocess.py) are stored.

The directory ``PSM/`` holds precalibrated statistical PSM files created by
LMR_PSMbuild.py.  Anytime proxy databases are updated, or adjustments to
statistical calibration are made, the files in this folder should be updated.

Again, if necessary, the path to most of these files can be directly specified in the
configuration file, but we recommend using the default directory structure.




