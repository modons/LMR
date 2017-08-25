#
# process results from climate_indices.py
#

import numpy as np
from functools import reduce

#production_cru_ccsm4_pagesall_0.75
#El Nino years sorted:
en_cc = [1195,1210,1255,1271,1310,1314,1330,1337,1368,1428,1431,1484,1485,1489,1490,1511,1537,1564,1565,1594,1617,1634,1635,1658,1681,1687,1710,1719,1747,1766,1770,1792,1793,1816,1838,1839,1858,1878,1884,1885,1900,1914,1915,1930,1931,1940,1941,1942,1969,1979]
#La Nina years sorted:
ln_cc = [1153,1217,1218,1233,1278,1288,1323,1324,1355,1363,1389,1399,1415,1416,1445,1464,1475,1495,1542,1552,1624,1625,1641,1670,1696,1697,1698,1733,1773,1781,1806,1880,1887,1890,1893,1894,1909,1910,1911,1917,1918,1925,1946,1954,1955,1956,1971,1972,1974,1976]

# production_cru_mpi_pagesall_0.75
#El Nino years sorted:
en_cm = [1145,1185,1310,1330,1337,1368,1404,1431,1435,1484,1485,1489,1490,1511,1519,1520,1527,1564,1565,1594,1615,1617,1634,1635,1639,1651,1687,1710,1719,1720,1747,1766,1792,1796,1804,1837,1838,1839,1859,1878,1879,1884,1885,1900,1931,1941,1942,1945,1947,1969]
#La Nina years sorted:
ln_cm = [1217,1218,1219,1233,1234,1323,1324,1340,1355,1399,1423,1445,1446,1542,1552,1581,1601,1602,1603,1622,1625,1626,1641,1662,1664,1697,1698,1717,1742,1781,1782,1801,1806,1830,1863,1880,1887,1890,1893,1894,1910,1911,1917,1918,1955,1956,1971,1972,1974,1976]

#production_mlost_ccsm4_pagesall_0.75
#El Nino years sorted:
en_mc = [1205,1210,1231,1310,1330,1337,1351,1368,1435,1485,1489,1490,1519,1526,1537,1564,1595,1613,1615,1617,1634,1635,1658,1687,1726,1747,1762,1776,1792,1803,1804,1808,1816,1838,1839,1846,1878,1883,1884,1885,1900,1906,1915,1930,1931,1941,1942,1947,1960,1969]
#La Nina years sorted:
ln_mc = [1207,1217,1218,1219,1234,1323,1324,1340,1355,1362,1399,1423,1444,1445,1495,1522,1542,1581,1587,1601,1625,1626,1641,1670,1697,1752,1772,1781,1786,1806,1826,1880,1887,1890,1893,1894,1909,1910,1911,1917,1918,1951,1954,1955,1956,1971,1972,1974,1975,1976]

#production_mlost_mpi_pagesall_0.75
#El Nino years sorted:
en_mm  =[1082,1145,1185,1210,1225,1226,1230,1231,1255,1330,1337,1351,1435,1489,1519,1520,1521,1526,1536,1537,1564,1595,1615,1617,1635,1658,1687,1719,1776,1777,1779,1804,1808,1816,1837,1838,1839,1878,1879,1884,1885,1900,1901,1915,1931,1942,1945,1947,1960,1969]
#La Nina years sorted:
ln_mm = [1101,1203,1217,1218,1219,1220,1234,1323,1324,1340,1355,1399,1423,1445,1456,1479,1487,1542,1581,1602,1603,1625,1644,1698,1742,1752,1781,1788,1790,1806,1820,1826,1854,1880,1887,1890,1893,1894,1895,1910,1911,1917,1918,1955,1956,1971,1972,1974,1975,1976]

#production_gis_mpi_pagesall_0.75
#El Nino years sorted:
en_gm = [1145,1185,1200,1225,1226,1255,1257,1294,1330,1351,1406,1435,1519,1520,1521,1538,1559,1562,1564,1565,1595,1615,1617,1635,1657,1687,1688,1748,1776,1779,1804,1808,1838,1839,1840,1866,1868,1878,1879,1884,1885,1900,1901,1915,1931,1941,1942,1945,1947,1960]
#La Nina years sorted:
ln_gm = [1071,1101,1203,1219,1324,1340,1399,1423,1445,1456,1479,1487,1542,1576,1580,1581,1589,1602,1603,1644,1662,1670,1697,1698,1741,1752,1781,1788,1806,1826,1835,1854,1863,1876,1880,1887,1890,1893,1894,1910,1911,1918,1922,1955,1956,1971,1972,1974,1975,1976]

#production_gis_ccsm4_pagesall_0.75
#El Nino years sorted:
en_gc = [124,1145,1210,1221,1225,1226,1255,1307,1330,1337,1351,1368,1386,1435,1467,1484,1490,1537,1538,1564,1565,1595,1617,1635,1657,1687,1688,1726,1747,1762,1766,1792,1803,1804,1824,1838,1846,1877,1878,1884,1885,1900,1915,1926,1930,1931,1941,1942,1945,1960]
#La,Nina years sorted:
ln_gc = [ 940,1101,1219,1233,1234,1262,1288,1323,1355,1363,1389,1399,1423,1456,1474,1479,1522,1542,1589,1625,1662,1670,1697,1752,1754,1781,1806,1811,1849,1863,1880,1887,1890,1893,1894,1909,1910,1911,1917,1918,1933,1951,1954,1955,1956,1971,1972,1974,1975,1976]

#
# El Nino
#

# common elements between two arrays
com = np.intersect1d(en_cc,en_mc)
print com

# common elements among many arrays
all = reduce(np.intersect1d,(en_cc,en_cm,en_mc,en_mm,en_gm,en_gc))
print all

#en_all = [en_cc,en_cm,en_mc,en_mm,en_gm,en_gc]
en_all = np.concatenate((en_cc,en_cm,en_mc,en_mm,en_gm,en_gc))

nds = len(en_all)
print 'number of recons:'+str(nds)

ec = np.zeros([nds,nds])
i = -1
for s1 in en_all:
    i = i + 1
    j = -1
    for s2 in en_all:
        j = j + 1
        com = np.intersect1d(s1,s2)
        ec[i,j] = len(com)

un,unc = np.unique(en_all,return_counts=True)

print 'total number of unique years:' + str(len(unc))

crit = 6
next = False
ml = []
for k in range(len(un)):
#    print str(un[k])+':'+str(unc[k])
    if k<len(un)-1 and un[k+1] == un[k] + 1:
        next = True
        es  = unc[k]+unc[k+1]
    else:
        next = False

    if unc[k] >= crit:
        print str(un[k])+':'+str(unc[k])
        ml.append(un[k])
    elif next and es >= crit and (unc[k+1]<crit):
        print str(un[k])+'+'+str(un[k+1])+'='+str(unc[k])+'+'+str(unc[k+1])
        if unc[k]>unc[k+1]:
            ml.append(un[k])
        else:
            ml.append(un[k+1])

print 'El Nino events:'
print ml
print len(ml)

#
# La Nina
#

# common elements among many arrays
ln_all = np.concatenate((ln_cc,ln_cm,ln_mc,ln_mm,ln_gm,ln_gc))

nds = len(ln_all)
print 'number of recons:'+str(nds)

ec = np.zeros([nds,nds])
i = -1
for s1 in en_all:
    i = i + 1
    j = -1
    for s2 in en_all:
        j = j + 1
        com = np.intersect1d(s1,s2)
        ec[i,j] = len(com)


un,unc = np.unique(ln_all,return_counts=True)

print 'total number of unique years:' + str(len(unc))

crit = 6
next = False
ml = []
for k in range(len(un)):
    if k<len(un)-1 and un[k+1] == un[k] + 1:
        next = True
        es  = unc[k]+unc[k+1]
    else:
        next = False

    if unc[k] >= crit:
        print str(un[k])+':'+str(unc[k])
        ml.append(un[k])
    elif next and es >= crit and (unc[k+1]<crit):
        print str(un[k])+'+'+str(un[k+1])+'='+str(unc[k])+'+'+str(unc[k+1])
        if unc[k]>unc[k+1]:
            ml.append(un[k])
        else:
            ml.append(un[k+1])

print 'La Nina events:'
print ml
print len(ml)

