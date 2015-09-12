import os
import h5py
import numpy as np

def split_data(path, split_size):
    """
    Split the datasets in an HDF5 file into smaller sets
    and save them to new files.
    """
    prefix = os.path.splitext(path)[0]
    f = h5py.File(path)
    n = 0
    # Find the largest n.
    for k,v in f.iteritems():
        n = max(n, v.value.shape[0])
    
    # Copy subsequences of the data to smaller files.
    width = int(np.ceil(np.log10(n / split_size)))
    for i,j in enumerate(range(0, n, split_size)):
        outfile = '{file}-{num:{fill}{width}}.h5'.format(
                file=prefix, num=i, fill='0', width=width)
        print(outfile)
        fout = h5py.File(outfile, 'w')
        for k,v in f.iteritems():
            subset = v[j:j+split_size]
            fout.create_dataset(k, data=subset, dtype=v.dtype)
        fout.close()
