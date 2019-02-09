epochs=10
batch=16

for eta in 0.01 0.05 1 1.5 2
do
	for lambd in 0.5 1 1.5 2 2.5 3 3.5
	do
		for alpha in 0 5 10 15 20 25
		do
			for beta in 0 5 10 15 20 25
			do
				python3 test.py $epochs $batch $eta $lambd $alpha $beta
			done
		done
	done
done
