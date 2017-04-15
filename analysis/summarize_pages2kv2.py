#================================================================================
# This script reads in the preprocessed PAGES2kv2 datasets and a PSM file and
# does two things:
#  1) Produces a list of all proxy types and units.
#  2) Produces individual figures of all 692 proxy records, along with metadata.
#
# Note: You'll need to make the preprocessed and PSM files before using this.
# Also, change the "data_directory" and "output_directory" to point to the
# appropriate places on your machine.
#
#    author: Michael P. Erb
#    date  : 2/13/2017
#================================================================================

import numpy as np
import matplotlib.pyplot as plt

### LOAD DATA

data_directory = "/home/scec-00/lmr/erbm/LMR/"
output_directory = "/home/scec-00/lmr/erbm/analysis/results/LMR/pages2kv2/figures/"
save_instead_of_plot = True

# Load the pages2kv2 proxy data and metadata as dataframes.
proxies_pages2k = np.load(data_directory+'data/proxies/NCDC_Pages2kv2_Proxies.df.pckl')
metadata_pages2k = np.load(data_directory+'data/proxies/NCDC_Pages2kv2_Metadata.df.pckl')

# Load an LMR PSM file.
psms_pages2k = np.load(data_directory+'PSM/PSMs_NCDC_Pages2kv2_annual_GISTEMP.pckl')


### CALCULATIONS

# Count all of the different proxy types and print a list.
archive_counts = {}
archive_types = np.unique(metadata_pages2k['Archive type'])
for type in archive_types:
    archive_counts[type] = np.unique(metadata_pages2k['Proxy measurement'][metadata_pages2k['Archive type'] == type],return_counts=True)

print "================"
print " Archive counts"
print "================"
for type in archive_types:
    for units in range(0,len(archive_counts[type][0])):
        print('%25s - %23s : %3d' % (type, archive_counts[type][0][units], archive_counts[type][1][units]))

# Save a list of all records with PSMs.
records_with_psms = []
for i in range(0,len(psms_pages2k)):
    records_with_psms.append(psms_pages2k.keys()[i][1])
        
        
### FIGURES
plt.style.use('ggplot')

#for i in range(0,3):  # To make sample figures, use this line instead of the next line.
for i in range(0,len(metadata_pages2k['NCDC ID'])):
    print "Proxy: ",i+1,"/",len(metadata_pages2k['NCDC ID'])
    if metadata_pages2k['NCDC ID'][i] in records_with_psms: has_psm = "YES"
    else: has_psm = "NO"
    #
    # Make a plot of each proxy.
    plt.figure(figsize=(10,8))
    ax = plt.axes([.1,.6,.8,.3])
    plt.plot(proxies_pages2k[metadata_pages2k['NCDC ID'][i]],'bo-')
    plt.title("Pages2k v2 proxy time series\n"+metadata_pages2k['NCDC ID'][i])
    plt.xlabel("Year")
    plt.ylabel(metadata_pages2k['Proxy measurement'][i])
    #
    # Print metadata on each figure.
    for offset, key in enumerate(metadata_pages2k):
        plt.text(0,-.3-.1*offset,key+":",transform=ax.transAxes)
        plt.text(.3,-.3-.1*offset,metadata_pages2k[key][i],transform=ax.transAxes)
    plt.text(0,-.4-.1*offset,"LMR can compute GISTEMP PSM:",transform=ax.transAxes)
    plt.text(.3,-.4-.1*offset,has_psm,transform=ax.transAxes)
    if save_instead_of_plot:
        plt.savefig(output_directory+metadata_pages2k['Archive type'][i].replace(" ","_")+"_"+metadata_pages2k['NCDC ID'][i]+"_record"+str(i)+".jpg")
    else:
        plt.show()
    plt.close()

