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
        # created final weighted hostlib iteration
        echo python3 hostlib_generator.py "$i" 3 C11
        python3 hostlib_generator.py "$i" 3 C11
        mv -v hostlib/"$i"/*GRID.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
        # run snlc_sim.exe
        sim_SNmix.pl SIMGEN_MASTER_"$i"_GRID.INPUT
        # run split_and_fit.pl
        split_and_fit.pl SIMFIT_"$i"_GRID.nml
        # move fitres file 1 from .nml outdir to fitres directory
        cp SIMFIT_"$i"_GRID/JSH16_G10_"$i"_GRID/FITOPT000.FITRES /project/rkessler/jaredhand/$hd/fitres/"$i"/FITOPT000_"$i".FITRES
        # run SALT2mu.exe on fitres file
        SALT2mu.exe SALT2mu_"$i".default file=FITOPT000_"$i".FITRES prefix=JSH_WFIRST_"$i"_mures
    done
