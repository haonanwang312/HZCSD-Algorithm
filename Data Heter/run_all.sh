# chmod +x run_all.sh
# ./run_all.sh
#!/bin/bash


for script in  1dig0.py 1dig1.py 1dig2.py 1dig3.py 1dig4.py 2dig0.py 2dig1.py 2dig2.py 2dig3.py 2dig4.py 3dig0.py 3dig1.py 3dig2.py 3dig3.py 3dig4.py 4dig0.py 4dig1.py 4dig2.py 4dig3.py 4dig4.py 5dig0.py 5dig1.py 5dig2.py 5dig3.py 5dig4.py 6dig0.py 6dig1.py 6dig2.py 6dig3.py 6dig4.py 7dig0.py 7dig1.py 7dig2.py 7dig3.py 7dig4.py 8dig0.py 8dig1.py 8dig2.py 8dig3.py 8dig4.py 9dig0.py 9dig1.py 9dig2.py 9dig3.py 9dig4.py 10dig.py 
do
    echo "==== Running $script ===="
    python3 $script
done
