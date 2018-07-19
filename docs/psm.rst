.. _psm:

Adding PSMs
===========
There are three different types of proxy system models (PSM) included in the
LMR: linear (temperature), linear T or P (temperature or precipitation) and
bilinear (temperature and precipitation). If you would like to add a different
PSM, follow the instructions below.

How to add a new PSM to the LMR
-------------------------------

1. Edit LMR_psms.py to make the new PSM class. Once the class is added, it must
   become part of the _psm_classes list.
2. Add the new PSM information to LMR_utils.py by generating a PSM calibration
   string and finding pkind.
3. Add the new PSM information to misc/build_ye_file.py by giving it a psm_key.
   There may be one or two other places to add information depending on what the
   PSM looks like.
4. Edit LMR_config.py and config.yml to include the new PSM class. Once the
   class is added, it must be initialized with def __init__.

