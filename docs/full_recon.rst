.. _full_recon:

********************************
Performing a full reconstruction
********************************

This section gives a brief walkthrough of the necessary steps for running a full
LMR reconstruction including: configuration, setting up proxies, PSMs, and
usage of pre-calculated observations.

Building the proxy database
===========================

.. note:: The provided databases in the downloaded :ref:`sample data <sample_data>`
  should be sufficient for most reconstruction purposes. Skip over this step
  unless updates were made to the available proxy data.

The proxy files used by the LMR code are in the format of Pandas dataframes
which are used as a database-like structure.  One file contains the list of
proxies with unique identifiers and associated metadata
(``xxxx_Metadata.df.pckl``), while the other contains the proxy measurements
over time (``xxxx_Proxy.df.pckl``).

Anytime proxies are added or updated the database files
need to be updated using ``LMR_proxy_preprocess.py``.  There are more
available options described in the comments of the ``main()`` function, but
we discuss the important parameters here.

The main choice is which proxy database to build.  There are two choices of
``proxy_data_source``, 'LMRdb' and 'PAGES2Kv1'.  'LMRdb' is a compilation of
NCDC, PAGES2K Phase 2, and other collected proxy records, while `PAGES2Kv1'
is data from only the PAGES2K Phase 1 project.  For most purposes, 'LMRdb'
should be the database of choice.  Comment out whichever source is not in use
e.g,::

    #proxy_data_source = 'PAGES2Kv1' # proxies from PAGES2k phase 1 (2013)

    # --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** ---

    proxy_data_source = 'LMRdb'     # proxies from PAGES2k phase 2 (2017) +
                                    # "in-house" collection in NCDC-templated files

The other important parameters are to set the proxy input data and database
output locations.  If you downloaded sample data from us, the input data are
located under the ``proxies`` directory (e.g.,
``/home/disk/foo/LMR_data/data/proxies/``).  It is fine to use that same
directory as the output target.

.. note:: If rebuilding 'LMRdb', you should make sure the files under
 ``.../LMR_data/proxies/LMRdb/`` are untarred first.  ``cd`` to that
 directory and untar using ``$ tar -xvf ToPandas_v0.4.0_files.tar.gz``.

When finished editing the options defining the database and file locations
create the databases using::

    (lmr_py3) $ python LMR_proxy_preprocess.py

(Remember to activate the correct python environment if you're using Anaconda!)

Create pre-calibrated statistical PSMs
======================================
.. note:: The provided sample data includes many pre-calibration combinations
 for seasonal/annual PSMs calibrated against GISTEMP, NOAA, Berkeley Earth,
 and GPCC.  You will likely be able to skip this step if no changes have
 been made to the underlying proxy database. If a needed pre-calibration file
 does not exist, the code alerts you that it was not found and exits.  At
 that point, follow the instructions in this section.

The proxy system models (PSMs) are essential for translating our
reconstructed fields (in climate model space) to something comparable to the
proxy data (observation space).  We implement a few different statistical
regressions that fit proxies against instrumental data to form a PSM.  These
models are fit using annual or seasonal averages and a
univariate or bivariate fit to moisture and temperature variables.

The ``LMR_PSMbuild.py`` script creates the pickle files found in the
``LMR_data/PSM/`` directory.  This file still uses a legacy
configuration style, so at first glance it's a bit more dense than other
configuration interactions. The parameters for users are denoted between the
makers::

    ##** BEGIN User Parameters **##

    ...

    ##** END User Parameters **##

Below, user parameters are described for each configuration section of
``LMR_PSMbuild.py``.  After setting the relevant parameters for desired PSM
calibration, create the files using the command::

    (lmr_py3) $ python LMR_PSMbuild.py

class v_core
------------

* **lmr_path**: Path to LMR input data folders (e.g., /home/disk/foo/LMR_data/)
* **psm_type**: Setting to use 'linear' or 'bilinear' statistical PSM
* **anom_reference_period**: The time period over which the average is
  taken to use as the reference value for all proxy PSMs
* **calib_period**: Years over which proxy and instrumental data are used to
  calibrate the PSM

class v_proxies
---------------

* **use_from**: Which proxy database to use for calibration (['PAGES2kv1'] or
  ['LMRdb'])

class v_psm
-----------

* **avgPeriod**: Whether to use annual or seasonal averages to calibrate the PSM
  ('annual' or 'season')
* **test_proxy_seasonality**: A flag where if True will go through a
  pre-defined set of seasonal distinctions to find the best calibration fit.
  The seasons tested are defined for each database for various proxy types.
  (Starting on Line #264 for PAGES2kv1 or Line #506 for LMRdb)

class _linear
^^^^^^^^^^^^^
* **datadir_calib**: Directory for instrumental calibration data. 'None'
  defaults to files in the designated lmr_path directory.
* **datatag_calib** and **datafile_calib**: Instrumental target for
  calibration.  Uncomment the pair for the desired data, and make sure all
  others are commented out.
* **psm_r_crit**: Correlation threshold to consider for PSM calibration. If
  a fit is below this threshold the PSM is not created for that proxy.

class_bilinear
^^^^^^^^^^^^^^

* **datadir_calib**: Directory for instrumental calibration data. 'None'
  defaults to files in the designated lmr_path directory.
* **datatag_calib_T** and **datafile_calib_T**: Instrumental target for
  temperature-sensitive calibration.  Uncomment the pair for the desired data,
  and make sure all others are commented out.
* **datatag_calib_P** and **datafile_calib_P**: Instrumental target for
  moisture-sensitive calibration.  Uncomment the pair for the desired data,
  and make sure all others are commented out.
* **psm_r_crit**: Correlation threshold to consider for PSM calibration. If
  a fit is below this threshold the PSM is not created for that proxy.


Configuring the LMR reconstruction
==================================

To start off, the configuration files need to be copied into the main source
code directory for LMR.  Wherever you cloned/downloaded the source code
(we’ll use the path /home/disk/foo/LMR_src for our code directory) there should
be a ``config_templs/`` folder which holds configuration templates.
From the LMR_src directory, there are two files you need to copy to
run an experiment::

    $ cp config_templs/config_template.yml ./config.yml
    $ cp config_templs/LMR_config_template.py ./LMR_config.py

The file, ``config.yml``, contains all the necessary knobs to fine-tune the
reconstruction.  For an explicit description of each option, please see
:ref:`configuration`.

Important options for a reconstruction
--------------------------------------

* **core**

  * **nexp**: Experiment name
  * **lmr_path**: Path to LMR_data directory
  * **datadir_output**: Working directory to temporarily store LMR output files
  * **archive_dir**: Archive directory to store final post-processed LMR output
  * **recon_period**: Range of years (edge inclusive) to reconstruct
  * **nens**: Number of prior ensemble members (should generally be above 50)
  * **save_archive**: Ensemble detail of field output. 'ens_variance' and
    'ens_percentiles' are more econmical reductions, while 'ens_subsample' and
    'ens_full' store full-field ensemble members and can use large amounts of
    disk space
  * **seed**: Sets the RNG seed to ensure reproducability for the ensemble
    sample and proxy record sample.  WARNING: overwritten by wrapper.multi_seed
    and should not be used when running multiple iterations of a
    reconstruction.

* **proxies**

  * **use_from**: Which proxy database to use for the reconstruction. [LMRdb]
    or [PAGES2kv1]
  * **proxy_frac**: Fraction of available proxy records to use. Useful for
    independent verification on withheld proxies
  * **proxy_order** (Database specific): Order of assimilation for proxy
    records. Commenting out proxy groups here will omit them from use in the
    reconstruction
  * **proxy_psm_type** (Database specific): Specifies which PSM type to be
    used for which proxy groups. E.g., Tree ring_Width: bilinear
  * (database specific means there are separate configuration settings for
    each proxy database)

* **psm**

  * **calib_period**: Distinction of instrumental time period to calibrate PSMs to
  * **avgPeriod**: Whether to use annual or seasonal averages to calibrate PSMs
  * **season_source** (Only used for seasonal PSMs): Use season defined by
    the proxy metadata or an objectively-derived best season
  * **datatag_calib** (PSM dependent): Which instrumental data source to use
    for calibration. Options defined in ``all_calib_sources`` parameter

* **prior**

  * **prior_source**: Experiment tag to use as source data for the prior
    ensemble.  Should match the tag defined in datasets.yml
  * **state_variables**: Which state variables to reconstruct and output.  If
    not using pre-calculated Ye-values (estimated observations) the
    PSM-required-variables must be listed (i.e., temperature and/or moisture
    fields).  The associated value after each field can be either 'anom' or
    'full'.  'anom' uses anomaly values for the
    prior. 'full' uses original non-centered data for the prior and is not
    guaranteed to work in all cases.
  * **regrid_method**: Specification for regridding data that is loaded in
    for the prior.  `esmpy` is generally recommended and can handle
    masked/non-regular grids.
  * **regrid_resolution** (simple or spherical harmonics only): Resolution of
    the regridded field. (Number is a reference to the spherical harmonics
    truncation.  E.g., 42 is a 44x66 grid.)
  * **esmpy_interp_method** (esmpy only): Which interpolation method to use
    ('bilinear' or 'patch')
  * **esmpy_regrid_to** (esmpy only): Target regrid definition tag defined
    in ``grid_def.yml``

Important options for a Monte-Carlo (MC) iteration
--------------------------------------------------

Advantages of the LMR framework include the capacity to run many realizations
of a reconstruction by sampling from the input data.  This generates uncertainty
bounds on reconstructed output and is an essential product for determining the
robustness of reconstructed signals.  There are a few options in the
configuration important for MC operations.

* **wrapper**

  * **iter_range**: Number range to perform iterations over.  [0, 5] will
    output reconstructions to 5 different directories named r0 - r5.  One
    can easily distribute
    runs on an a cluster by farming out different iteration ranges. E.g., set
    the range as [0, 5] for a reconstruction on one machine and [6, 10] on another.
  * **multi_seed**: Seeds for creating reproducible iterations.  Must be of
    length such that indexing from the ``iter_range`` number is not out of
    bounds.

* **core**

  * **nens**: Number of prior ensembe members (should generally be above 50).
    This is resampled for each iteration.

* **proxies**

  * **proxy_frac**: Fraction of available proxy records to use. Useful for
    independent verification on withheld proxies. Resampled for each iteration.

Pre-calculating estimated observations (Ye values)
==================================================

For offline reconstructions, estimated observations from the prior sample are
re-used each year.  If we are not interested in outputting a field required for
the PSM, the Ye values (estimated observations) can be calculated and the field
ommitted. This saves memory and disk space and allows for individual fields
to be reconstructed separately when using RNG seeding (i.e., ``multi_seed`` for
MC reconstructions).

To enable this, we first need to create the pre-calculated Ye file.  After
setting up the config.yml, ``cd`` into the ``misc/`` directory and run
the command::

    (lmr_py3) $ python build_ye_file.py <path/to/desired/config.yml>

If no configuration file is provided as a command-line argument, the code
uses ``config.yml`` in the source code directory. This script builds the Ye file
based on the chosen proxy database, PSMs and averaging period, and the prior
source.  Numpy zip files contining the calculated Ye values are output in the
``lmr_path`` directory under ``ye_precalc_files``.

In order to use the file of pre-calculated Ye values, in ``config.yml`` under
the ``core`` section, set ``use_precalc_ye`` to True.

Running your LMR reconstruction
===============================

With all the files created and the configuration set, running a reconstruction
is performed using::

    (lmr_py3) $ python LMR_wrapper.py

If any files are missing or the configuration is set up incorrectly, the code
will exit with an error printout explaining what action should be taken.

After the reconstruction finishes, there is a printout of total time
elapsed, and the code issues a move command to process the output and place it
in the designated archive directory.



