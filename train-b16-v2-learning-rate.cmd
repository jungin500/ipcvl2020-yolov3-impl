@ECHO OFF

echo Training with Learning rate 5e-4
python main.py --manifest-train manifest-16-v2.txt -b 16 -m -l 5e-4 -e 400

echo Training with Learning rate 1e-3
python main.py --manifest-train manifest-16-v2.txt -b 16 -m -l 1e-3 -e 400

echo Training with Learning rate 5e-3
python main.py --manifest-train manifest-16-v2.txt -b 16 -m -l 5e-3 -e 400