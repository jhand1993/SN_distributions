#! /bin/sh

# unload and load python for midway
module unload python
module load python/3.4-2015q1
# home directory
hd=SIMS
ms=1KMS0

# combine fitres with no mures data, run SALT2mu on combined fitres
echo python3 combine_nomures.py $ms
python3 combine_nomures.py $ms
SALT2mu.exe SALT2mu_combine.default file=fitres/combine/combine_$ms.fitres prefix=combine_"$ms"_mures u1=0 u2=0 u7=1 u8=1
mv combine_"$ms"_mures.fitres fitres/combine/
