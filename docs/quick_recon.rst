.. _quick_recon:

Quickstart reconstruction
==========================

After getting Python installed and downloading the LMR sample data, you are now
set to run a simple reconstruction of surface air temperature!

In this section we’ll start with the ``LMR_lite.py`` script which runs a
reconstruction but hides many of the details.  In this section we’ll run through
a simple reconstruction that uses the downloaded :ref:`sample data <sample_data>`.

.. note:: Many of the utility functions used in this script and in
  ``LMR_lite_utils.py`` are useful for running LMR in jupyter notebook
  environments, which promotes rapid protyping with data stored in memory.

Configuration
-------------

To start off, the configuration files need to be copied into the main source
code directory for LMR.  Wherever you cloned/downloaded the source code 
(we’ll use the path /home/disk/foo/LMR_src for our code directory) there should
be a ``config_templs/`` folder which holds configuration templates.
From the LMR_src directory, there are two files you need to copy to
there to perform an experiment::

    $ cp config_templs/config_lite_template.yml ./config_lite.yml
    $ cp config_templs/LMR_config_template.py ./LMR_config.py

The ``config_lite.yml`` file holds a subset of the configuration parameters we’ll
need for the reconstruction, while ``LMR_config.py`` is the main file for all
configuration business logic.

Next, you’ll have to edit a line in ``config_lite.yml``. In this file, edit the
path ``lmr_path`` defined in the ``core`` section.  It should be changed to
point to the input data folder (e.g., /home/disk/foo/LMR_data or whatever you
defined when unzipping the sample data) ::

    core:
      nexp: test_lmr_recon
      lmr_path: /home/path/to/LMR/

.. note:: There are many more options that allow for fine control over a
  reconstruction. Please see :ref:`configuration` for details. The sample files
  provided let you run this experiment out of the box. Other configuration
  changes may require you to recreate intermediate files associated with PSM calibration
  and estimated proxy values (Ye values) from the prior (climate model data).

Running LMR_lite
----------------

After making the configuration change the reconstruction is ready for launch!

If you installed an Anaconda environment, make sure that you have the correct
one activated. E.g., ::

    $ source activate lmr_py3

Then you can run a reconstruction using::

    $ (lmr_py3) python LMR_lite.py

The code will print out reconstruction progress in your terminal.
After the reconstruction is finished you’ll see the total
time elapsed, a few plot windows will open up if you are running locally, and
you'll see that a file of analyzed output will be saved in the source directory
(analyses_1900_2000_1880_2000.npz). And that’s all for a simple
reconstruction!  For an in-depth
description of configuring and running a reconstruction from start to finish
please see :ref:`full_recon`.

Example end of reconstruction output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    1980: gmt=    0.364120 nhmt=    0.315032 shmt=    0.414933
    1981: gmt=    0.215862 nhmt=    0.181326 shmt=    0.251612
    1982: gmt=    0.142075 nhmt=    0.119087 shmt=    0.165872
    1983: gmt=    0.196176 nhmt=    0.187506 shmt=    0.205150
    1984: gmt=    0.310819 nhmt=    0.344414 shmt=    0.276043
    1985: gmt=    0.221263 nhmt=    0.284358 shmt=    0.155951
    1986: gmt=    0.403800 nhmt=    0.503617 shmt=    0.300477
    1987: gmt=    0.397144 nhmt=    0.444159 shmt=    0.348477
    1988: gmt=    0.540332 nhmt=    0.636867 shmt=    0.440407
    1989: gmt=    0.345345 nhmt=    0.314784 shmt=    0.376979
    1990: gmt=    0.555479 nhmt=    0.623369 shmt=    0.485203
    1991: gmt=    0.590704 nhmt=    0.662194 shmt=    0.516702
    1992: gmt=    0.332070 nhmt=    0.360107 shmt=    0.303047
    1993: gmt=    0.396883 nhmt=    0.546958 shmt=    0.241537
    1994: gmt=    0.595685 nhmt=    0.762036 shmt=    0.423490
    1995: gmt=    0.537395 nhmt=    0.650302 shmt=    0.420523
    1996: gmt=    0.455852 nhmt=    0.482466 shmt=    0.428303
    1997: gmt=    0.479458 nhmt=    0.575316 shmt=    0.380233
    1998: gmt=    0.603113 nhmt=    0.718212 shmt=    0.483970
    1999: gmt=    0.424851 nhmt=    0.564871 shmt=    0.279912
    analyses_1900_2000_1880_2000.npz exists...loading it
    returning global means...
    failed to get the current screen resources
    saving to .png

    -----------------------------------------------------
    Reconstruction completed in 9.480862776438395 mins
    -----------------------------------------------------

Figure of global mean temperatures against analysis data produced by
``LMR_lite.py``.

.. image:: ../lite_testing_GMT_annual.png




