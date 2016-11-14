#! /bin/sh

# unload and load python for midway
module unload python
module load python/3.4-2015q1
# survey names.  Must be upper case.
survs=(SDSS SNLS LOWZ)
# Iterate though respective hostlibs to create accurate weight maps.
for i in "${survs[@]}"
    do
        # create first iteration of hostlibs with flat mass distribution
        echo python3 hostlib_generator.py "$i" 0
        python3 hostlib_generator.py "$i" 0
        mv -v hostlib/"$i"/*.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
        # run snlc_sim.exe
        snlc_sim.exe jsh_SIMGEN_"$i"_G10.INPUT
        # run split_and_fit.pl
        split_and_fit.pl "$i".nml
        # move fitres file 1 from .nml outdir to fitres directory
        cp "$i"/JSH_"$i"_G10/FITOPT000.FITRES /project/rkessler/jaredhand/hostmass/fitres/"$i"/JSH_WFIRST_"$i"1.FITRES
        # create second iteration of hostlibs
        echo python3 hostlib_generator.py "$i" 1
        python3 hostlib_generator.py "$i" 1
        mv -v hostlib/"$i"/*.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
        # run snlc_sim.exe
        snlc_sim.exe jsh_SIMGEN_"$i"_G10.INPUT
        # run split_and_fit.pl
        split_and_fit.pl "$i".nml
        # move fitres file 1 from .nml outdir to fitres directory
        cp "$i"/JSH_"$i"_G10/FITOPT000.FITRES /project/rkessler/jaredhand/hostmass/fitres/"$i"/JSH_WFIRST_"$i"2.FITRES
        # create third iteration of hostlibs
        echo python3 hostlib_generator.py "$i" 2
        python3 hostlib_generator.py "$i" 2
        mv -v hostlib/"$i"/*.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
        # run snlc_sim.exe
        snlc_sim.exe jsh_SIMGEN_"$i"_G10.INPUT
        # run split_and_fit.pl
        split_and_fit.pl "$i".nml
        # move fitres file 1 from .nml outdir to fitres directory
        cp "$i"/JSH_"$i"_G10/FITOPT000.FITRES /project/rkessler/jaredhand/hostmass/fitres/"$i"/JSH_WFIRST_"$i"3.FITRES
    done
