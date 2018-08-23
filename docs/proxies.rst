.. _proxies:

Adding proxy records
====================

NCDC file formatting:
---------------------
[Example of correctly formatted file with additional information about possible
options.]

How to add proxy records to the LMR
-----------------------------------
1. Add a new class to LMR_config.py.
2. Add new proxy database information to LMR_proxy_preprocess.py into def main
   and create and new function to create the database from whatever starting
   files.
3. Add new proxy database information to LMR_proxy_pandas_rework.py by adding a
   class for the proxy database and adding the new database as an option to
   _proxy_classes.
4. In LMR_utils.py, add new proxy_str and proxy_cfg options for the new database.
5. In misc/build_ye_file.py, add new proxy_cfg option for the new database.
6. Update LMR_PSMbuild.py if you wish to use this utility.

