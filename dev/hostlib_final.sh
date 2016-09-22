#! /bin/sh

# survey names.  Must be upper case.
survs=(SDSS SNLS LOWZ)
for i in "${survs[@]}"
    do
        # created final weighted hostlib iteration
        echo python3 hostlib_generator.py "$i" 3
        python3 hostlib_generator.py "$i" 3
        mv -v hostlib/"$i"/*.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
        # run snlc_sim.exe
        snlc_sim.exe jsh_SIMGEN_"$i"_G10.INPUT
        # run split_and_fit.pl
        split_and_fit.pl "$i".nml
        # move fitres file 1 from .nml outdir to fitres directory
        cp "$i"/JSH_"$i"_G10/FITOPT000.FITRES /project/rkessler/jaredhand/hostmass/fitres/"$i"/FITOPT000_"$i".FITRES
        # run SALT2mu.exe on fitres file
        SALT2mu.exe SALT2mu_"$i".default file=FITOPT000_"$i".FITRES prefix=JSH_WFIRST_"$i"_mures
    done
# make unique directory for mures fitres files
TIME=$(date "+%m-%d-%y_%H_%M_%S")
mkdir -p fitres/$TIME
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
cp mures/combine mures/$TIME
