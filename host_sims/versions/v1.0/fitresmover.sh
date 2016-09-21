#! /bin/sh

# rename fitres files for use in combine.py
mv JSH_WFIRST_SDSS.fitres JSH_WFIRST_SDSS4_mures.fitres
mv JSH_WFIRST_SNLS.fitres JSH_WFIRST_SNLS4_mures.fitres
# mv JSH_WFIRST_PS1.fitres JSH_WFIRST_PS14_mures.fitres
mv JSH_WFIRST_LOWZ.fitres JSH_WFIRST_LOWZ4_mures.fitres

TIME=$(date "+%m-%d-%y_%H_%M_%S")

mkdir -p fitres/$TIME
# cp {JSH_WFIRST_SDSS4_mures.fitres,JSH_WFIRST_SNLS4_mures.fitres,JSH_WFIRST_PS14_mures.fitres,JSH_WFIRST_LOWZ4_mures.fitres} fitres/$TIME
cp {JSH_WFIRST_SDSS4_mures.fitres,JSH_WFIRST_SNLS4_mures.fitres,JSH_WFIRST_LOWZ4_mures.fitres} fitres/$TIME 
mv JSH_WFIRST_SDSS4_mures.fitres fitres/sdss/
mv JSH_WFIRST_SNLS4_mures.fitres fitres/snls/
# mv JSH_WFIRST_PS14_mures.fitres fitres/ps1/
mv JSH_WFIRST_LOWZ4_mures.fitres fitres/lowz/

python3 combine.py
python3 hubbleres_combine.py

cp fitres/composite/composite.fitres fitres/$TIME/
