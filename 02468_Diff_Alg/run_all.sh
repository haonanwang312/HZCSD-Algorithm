# chmod +x run_all.sh
# ./run_all.sh
#!/bin/bash


# for script in ZO_HICZ.py NoSign_HICZ.py 2bit_HIZC.py #Top150_HICZ.py Rand300_HICZ.py Top50_HICZ.py Top100_HICZ.py Rand200_HICZ.py Rand400_HICZ.py
for script in  Top150_HICZ.py Top100_HICZ.py Rand200_HICZ.py 
do
    echo "==== Running $script ===="
    python3 $script
done
