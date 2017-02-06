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
        echo python3 hostlib_generator.py "$i" 3 G10
        python3 hostlib_generator.py "$i" 3 G10
        mv -v hostlib/"$i"/*.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
        # run snlc_sim.exe
        sim_SNmix.pl SIMGEN_MASTER_"$i".INPUT
    done
