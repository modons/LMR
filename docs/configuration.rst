.. _configuration:

LMR Configuration
=================

.. toctree::
   :maxdepth: 2

Overview
--------
The LMR configuration groups a set of user defined parameters detailing
the reconstruction experiment including: the proxy data to use, the
fields to be reconstructed, and aspects of the data assimilation method.
The ``config_template.yml`` file should be copied into the source directory
as ``config.yml``.  This is the default file searched for by the code to run
a reconstruction and holds the parameters available to users.  Use cases are
described below followed by a general outline of the parameters available.


General Configuration
---------------------
When running a reconstruction, the ``LMR_wrapper.py`` script is set up to
look for ``config.yml`` in the code directory to use as the configuration.
This file is a YAML (
`YAML Ain't Markup Language <http://www.yaml.org/spec/1.2/spec.html>`_ ;
`useful primer <https://learn.getgrav.org/advanced/yaml>`_) document that
gets read in at runtime.  Each section of the file describes the user
parameters for a specific aspect of the reconstruction.  Casting of the
values from the file into python are done by the yaml parser, so when
editing the file **please try and maintain the same data type as the template**.

wrapper:
    Parameters related to orchestrating the reconstruction
    realizations.  I.e. Monte-Carlo iterations, parameter space searches.
core:
    High-level reconstruction parameters such as main data and
    output directories, experiment name, and DA controls.
proxies:
    Parameters controlling which proxy database to use, how the proxies are
    selected, and which observation models are used.
psms:
    Parameters for setting up and using different proxy observation models.
prior:
    Parameters describing data source and fields to use as the prior state
    estimate during a reconstruction.

.. note::
   If ``config.yml`` is not found or if any extraneous parameters (including misspellings)
   are found in the file, the reconstruction code will exit immediately.

Custom Configuration Files
--------------------------

If you would like to use a file other than ``config.yml`` as the reconstruction
configuration ``LMR_wrapper.py`` is set up to so the first runtime argument
can be passed as the configuration to use ::

    LMR_wrapper.py /path/to/a/different_config.yml

With this you might store common configurations somewhere else instead of constantly
changing ``config.yml``.

.. note::
    If the file specified as an argument is not found, the code will exit immediately.

Legacy Configuration
--------------------

The LMR code was originally set up to use ``LMR_config.py`` as the primary
configuration mechanism.  It provided an easy object-oriented way to
encaspulate parameters passed around to different classes at runtime.
The nature of providing parameter listings that couldn't be changed
during an experiment at by outside references to the configuration
reduced the readability.  To switch away from using the YAML files
just set the following flag at the top of ``LMR_config.py`` ::

    LEGACY_CONFIG = True

This means all parameters will be specified within ``LMR_config.py``
between the commented sections ::

    ##** BEGIN User Parameters **##
    parameter1 = True
    parameter2 = '/test_dir'
    ##** END User Parameters **##

Programmatic Config Updating
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some instances you may want to update configuration values on the fly.
There are a few different ways to accomplish this within ``LMR_config.py``.

A more permanent change which will be propagated to all subsequent ``Config``
instances can be accomplished by editing the values of a
class definition directly. ::

    LMR_config.core.nexp = 'New_experiment'
    LMR_config.core.nens = 20
    LMR_config.proxies.pages.datadir_proxy = '/new/path/to/proxy/data'

You can also permanently update the configuration using dictionaries much like those
imported from the YAML files. ::

    update_dict = {'core': {'nexp': 'New_experiment',
                            'nens': 20},
                   'proxies': {'pages': {'datadir_proxy': '/new/path/to/proxy/data'}}}
    LMR_config.update_config_class_yaml(update_dict, LMR_config)

If only temporary changes to the configuration are necessary, instead just pass
the dictionary of update key/value pairs to the constructor. ::

    update_dict = {'core': {'nexp': 'New_experiment',
                            'nens': 20},
                   'proxies': {'pages': {'datadir_proxy': '/new/path/to/proxy/data'}}}
    cfg = LMR_config.Config(**update_dict)

This will make no alterations to the imported ``LMR_config.py``.


Reference
---------

.. automodule:: LMR_config
   :members:
