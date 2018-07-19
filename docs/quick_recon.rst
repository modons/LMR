.. _quick_recon:

Quick Start Reconstruction
==========================

After getting Python installed and downloading the LMR sample data, you are now
set to run a simple reconstruction!

In this section we’ll start with the ``LMR_lite.py`` script which can run a full
reconstruction but hides many of the details.  In this section we’ll run through
a simple reconstruction that uses the base data downloaded in `sample_data`_.

.. note:: Many of the utility functions used in this script and in
``LMR_lite_utils.py`` are useful for running LMR in jupyter notebook
environments.

Configuration
-------------

To start off, the configuration files need to be copied into the main source
code directory for LMR.  Wherever, you cloned/downloaded the source code to
(we’ll use the path /home/disk/foo/LMR_src for our code directory) there should
be a ``config_templs/`` folder which holds configuration templates.  There are
two files we’ll need to copy, so from the source directory::

    $ cp config/confg_lite_template.yml ./config_lite.yml
    $ cp config/LMR_config_template.py ./LMR_config.py

The ``config_lite.yml`` holds a subset of the configuration parameters we’ll
need for the reconstruction, while ``LMR_config.py`` is the main file for all
configuration business logic.

Next, you’ll have to edit a few things in ``config_lite.yml``.

First, you’ll have to set the input data location, the working directory for
output, and where the eventual reconstruction will be archived. To do this, edit
the paths defined in the ``core`` section: ``lmr_path`` should point to the
input data folder (e.g., /home/disk/foo/LMR_data or whatever you defined when
unzipping the sample data), while ``datadir_output`` and ``archive_dir`` should
be directories for initial and long-term storage. ::


    core:
      nexp: test_lmr_recon
      lmr_path: /home/path/to/LMR/

      datadir_output: /home/path/to/working_output
      archive_dir: /home/path/to/archive_output

..note:: There are many more options allowing for fine control over a
reconstruction. Please see :ref:`configuration` for details. The sample files
provided let you run this experiment out of the box. Other configuration changes
may require you to recreate intermediate files associated with PSM calibration
and estimated obs (Ye values) from the prior.

Running LMR_lite
----------------

After making the configuration changes the reconstruction is ready for launch!

If you installed an Anaconda environment, make sure that you have the correct
one activated. E.g., ::

    $ source activate lmr_py3

Then you can run a reconstruction using::

    $ (lmr_py3) python LMR_lite.py

After this print-outs of reconstruction progress should begin to appear in your
terminal.  After the reconstruction is finished you’ll see a read out of total
time elapsed and commands moving the finished reconstruction files to the
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



