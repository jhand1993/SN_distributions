#! /bin/sh
survs=(SDSS SNLS LOWZ)
# create first iteration of hostlibs with flat mass distribution
for i in "${survs[@]}"
do
echo python3 hostlib_generator.py "$i" 0
python3 hostlib_generator.py "$i" 0
mv -v hostlib/"$i"/*.HOSTLIB /project/rkessler/SN/SNDATA_ROOT/simlib/
# run snlc_sim.exe
snlc_sim.exe jsh_SIMGEN_"$i"_G10.INPUT
# run split_and_fit.pl
split_and_fit.pl "$i".nml
done
