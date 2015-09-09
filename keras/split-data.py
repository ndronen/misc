import os
import h5py

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
    for i,j in enumerate(range(0, n, split_size)):
        outfile = '{0}-{1}.h5'.format(prefix, i)
        print(outfile)
        fout = h5py.File(outfile, 'w')
        for k,v in f.iteritems():
            subset = v[j:j+split_size]
            fout.create_dataset(k, data=subset, dtype=v.dtype)
        fout.close()
