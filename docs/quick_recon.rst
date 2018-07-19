.. _quick_recon:

Quickstart reconstruction
==========================

After getting Python installed and downloading the LMR sample data, you are now
set to run a simple reconstruction!

In this section we’ll start with the ``LMR_lite.py`` script which runs a full
reconstruction but hides many of the details.  In this section we’ll run through
a simple reconstruction that uses the base data downloaded in `sample_data`_.

.. todo: Make sure the sample data actually references the sample download
   specified in the installation


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

    $ cp config/confg_lite_template.yml ./config_lite.yml
    $ cp config/LMR_config_template.py ./LMR_config.py

The ``config_lite.yml`` file holds a subset of the configuration parameters we’ll
need for the reconstruction, while ``LMR_config.py`` is the main file for all
configuration business logic.

Next, you’ll have to edit a few things in ``config_lite.yml``.

First, you will set the input data location, the working directory for
output, and the location where the reconstruction will be archived when
it is completed. To do this, edit
the paths defined in the ``core`` section: ``lmr_path`` should point to the
input data folder (e.g., /home/disk/foo/LMR_data or whatever you defined when
unzipping the sample data), while ``datadir_output`` and ``archive_dir`` should
be directories for initial and long-term storage. ::


    core:
      nexp: test_lmr_recon
      lmr_path: /home/path/to/LMR/

      datadir_output: /home/path/to/working_output
      archive_dir: /home/path/to/archive_output

.. note:: There are many more options that allow for fine control over a
  reconstruction. Please see :ref:`configuration` for details. The sample files
  provided let you run this experiment out of the box. Other configuration
  changes may require you to recreate intermediate files associated with PSM calibration
  and estimated proxy values (Ye values) from the prior (climate model data).

Running LMR_lite
----------------

After making the configuration changes the reconstruction is ready for launch!

If you installed an Anaconda environment, make sure that you have the correct
one activated. E.g., ::

    $ source activate lmr_py3

Then you can run a reconstruction using::

    $ (lmr_py3) python LMR_lite.py

T code will print out reconstruction progress in your
terminal.  After the reconstruction is finished you’ll see the total
time elapsed and commands the code issues to move the finished reconstruction files to the
archive directory.  And that’s all for a simple reconstruction!  For an in-depth
description of configuring and running a reconstruction from start to finish
please see :ref:`full_recon`.

Example end of reconstruction output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    =====================================================
    Reconstruction completed in 1.1332964181900025 mins
    =====================================================
    ('saving global mean temperature update history and ', 'assimilated proxies...')

    =====================================================
    Experiment completed in 1.1385778466860452 mins
    =====================================================
    State variable: tas_sfc_Amon
    writing the new ensemble mean file.../home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/ensemble_mean_tas_sfc_Amon
    writing the new ensemble variance file.../home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/ensemble_variance_tas_sfc_Amon
    State variable: tos_sfc_Omon
    writing the new ensemble mean file.../home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/ensemble_mean_tos_sfc_Omon
    writing the new ensemble variance file.../home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/ensemble_variance_tos_sfc_Omon
     **** clean start --- removing existing files in iteration output directory
    mv -f /home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/*.npz /home/katabatic2/wperkins/LMR_output/testing/test_flexible_outputs/r0/
    mv -f /home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/*.h5 /home/katabatic2/wperkins/LMR_output/testing/test_flexible_outputs/r0/
    mv -f /home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/*.pckl /home/katabatic2/wperkins/LMR_output/testing/test_flexible_outputs/r0/
    mv: cannot stat '/home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/*.pckl': No such file or directory
    mv -f /home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/assim* /home/katabatic2/wperkins/LMR_output/testing/test_flexible_outputs/r0/
    mv -f /home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0/nonassim* /home/katabatic2/wperkins/LMR_output/testing/test_flexible_outputs/r0/
    rm -f -r /home/katabatic/wperkins/data/LMR/output/working/test_flexible_outputs/r0
    cp /home/disk/p/wperkins/Research/LMR/config.yml /home/katabatic2/wperkins/LMR_output/testing/test_flexible_outputs/r0/

    2018-07-18 15:21:02.649270



