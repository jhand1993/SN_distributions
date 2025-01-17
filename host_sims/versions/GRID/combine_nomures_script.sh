#! /bin/sh

# unload and load python for midway
module unload python
module load python/3.4-2015q1
# home directory
hd=SIMS
dir=1KMS0
ms=1KMS0

# combine fitres with no mures data, run SALT2mu on combined fitres
echo python3 combine_nomures.py $ms $dir
python3 combine_nomures.py $ms $dir
SALT2mu.exe SALT2mu_combine.default file=fitres/"$ms"_composite/"$ms"_composite_nomures.fitres prefix="$ms"_composite_mures u1=0 u2=0 u7=1 u8=1
mv "$ms"_composite_mures.* fitres/"$ms"_composite/
