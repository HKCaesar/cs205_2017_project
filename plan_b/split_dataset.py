import argparse
import pandas as pd
import numpy as np
import os
from utilities import generate_random_subset


description = \
"""
This function takes a dataset and splits it into subsets of the orignal data.
"""

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                    description=description,
                    epilog='',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-d",
                    "--dataset",
                    metavar='<dataset>',
                    type=str,
                    help="A dataset in csv format",
                    required=True)

parser.add_argument("-o",
                    "--output_folder",
                    metavar='<output folder>',
                    type=str,
                    help="The folder where the new datasets will be stored.",
                    required=True)

parser.add_argument("-c",
                    "--columns",
                    metavar='<columns>',
                    type=str,
                    help="A sequence of comma delimited columns")

parser.add_argument("-n",
                    "--n_splits",
                    metavar='<n_splits>',
                    type=int,
                    help="The number of splits needed of the data.",
                    default=10)
I
args = parser.parse_args()

filename = args.dataset
folder = args.output_folder

df = pd.read_csv(filename)

print("Subsetting %s ..." %filename)
for percent in np.linspace(0,1,args.n_splits+1)[1:]:
    random_subset_df = generate_random_subset(df,percent)

    basename = filename.rsplit('.',1)[0]
    ext      = filename.rsplit('.',1)[1]

    random_subset_filename = ".".join([basename,str(percent),ext])
    random_subset_filename = os.path.join(folder, random_subset_filename)

    if args.columns is not None:
        columns = list(map(lambda s: s.strip(),args.columns.split(',')))
        random_subset_df=random_subset_filename[ columns ]

    print("\t%s (%.0f" %(random_subset_filename, percent*100)  + "%)" )

    random_subset_df.to_csv(random_subset_filename,index=False,header=True)


