#! /bin/sh

# unload and load python for midway
module unload python
module load python/3.4-2015q1
# home directory
hd=SIMS
# survey names.  Must be upper case.
survs=(SDSS SNLS LOWZ)
ms=1KMS0
for i in "${survs[@]}"
    do
        # move fitres file 1 from .nml outdir to fitres directory
        cp SIMFIT_"$i"_"$ms"/JSH_"$ms"_G10_"$i"/FITOPT000.FITRES FITOPT000_"$i".FITRES
        # run SALT2mu.exe on fitres file
        SALT2mu.exe SALT2mu_"$i".default file=FITOPT000_"$i".FITRES prefix=JSH_WFIRST_"$i"_mures
    done
# make unique directory for mures fitres files and final analysis results
TIME=$(date "+%m-%d-%y_%H_%M_%S")
mkdir -p fitres/$TIME
mkdir -p mures_plots/$TIME
# rename and move final mures fitres files to TIME directory
for i in "${survs[@]}"
    do
        # tack on '4' to each finished fitre file
        mv JSH_WFIRST_"$i"_mures.fitres JSH_WFIRST_"$i"4_mures.fitres
        cp JSH_WFIRST_"$i"4_mures.fitres fitres/$Time
        mv JSH_WFIRST_"$i"4_mures.fitres fitres/"$i"
    done

# combine fitres files and run analysis/produce plots
echo python3 combine.py
python3 combine.py
echo python3 hubbleres_combine.py
python3 hubbleres_combine.py

# create copy of mures/combine directory named TIME
cp mures_plots/combine mures_plots/$TIME
