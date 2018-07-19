.. _full_recon:

*******************************
Performin a Full Reconstruction
*******************************

.. warning:: This section is a WIP

To run an experiment, you must edit configuration. First, in the LMR code directory,
copy the file ``LRM_config_template.py`` to ``LMR_config.py``.  In ``LMR_config.py``,
change the ``SRC_DIR`` variable to the LMR code directory on your machine (i.e. the
directory which contains the ``LMR_config_template.py`` file).

Next, copy ``config_template.yml``, which is under version control, to ``config .yml``,
which is not. ::

    cp config_template.yml config.yml

Now edit ``config.yml``. At a minimum, do the following:

.. The existence requirement below should be verified [THIS IS A COMMENT]

1. set ``core.lmr_path`` to the directory containing the proxy, prior, and calibration data for the experiment (see details below). If you are at UW use R. Tardif's path as set by default in the config file. Otherwise, use the
   /home/disk/foo/LMR path where you unpacked the tar file in the steps described above.

2. set ``core.datadir_output`` to the working directory output for LMR. This is where data will be written during the experiment, so local disk is better (faster) than over a network. If the directory does not exist, you need to create it before running an experiment.

3. set ``core.archive_dir`` to the LMR reconstruction archive directory. This is where the experiment is archived, and datadir_output is scrubbed clean (frees up local disk space; often the archive is located across a network, but speed is no longer an issue). Note that data is lost in this step (ensemble mean from full ensemble), which you may wish to change at some point. Again, if the directory does not exist, you need to create it before running an experiment.

You are now ready to run an experiment from the code directory! ::

    python LMR_wrapper.py


..  note::  More steps are involved to use the NCDC proxy database,
pre-calculated obs estimate files, and certain psm pre-calibration files that will speed up the reconstruction process. [PLACE LINKS TO DOCUMENTATION PAGES IN THIS NOTE]


