#! /bin/sh

# unload and load python for midway
module unload python
module load python/3.4-2015q1
# home directory
hd=SIMS
# survey names.  Must be upper case.
survs=(SDSS SNLS LOWZ)
for i in "${survs[@]}"
    do
        # run split_and_fit.pl
        split_and_fit.pl SIMFIT_"$i".nml
    done
