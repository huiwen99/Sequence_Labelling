
# Instructions to run each script

## Requirements for running
1. Python 3.8
2. Pandas library
3. Numpy library

## Code to run each part
### Part 2
To train the emission parameters, run the following:
```python3 part2.py```

The emission parameters will be saved as a pickle file (```params.pkl```) in the respective dataset folder.

To predict with the trained parameters, run the following:
```python3 part2.py```

The output file is saved as ```dev.p2.out``` in the respective dataset folder.

### Part 3
Part 3 requires the emission parameters from part 2 (which is saved as a pickle file after training).

To train the transition parameters, run the following:
```python3 part3.py```

The transition parameters will be saved as a pickle file (```y_params.pkl```) in the respective dataset folder.

To predict with the trained parameters, run the following:
```python3 part3.py```

The output file is saved as ```dev.p3.out``` in the respective dataset folder.

### Part 4
Part 4 requires the emission parameters from part 2 and the transition parameters from part 3 (which is saved as a pickle file after training).

To convert the parameters to their respective dictionaries, run the following:
```python3 part4.py```

The dictionary for emission parameters is saved as ```em_dic.p``` and the dictionary for transition parameters is saved as ```tr_dic.p```.

To output the 3rd best sequence, run the following:
```python3 part4.py```

The output file is saved as ```dev.p4.out``` in the EN folder.


### Part 5
Part 5 requires the emission parameters from part 2 and the transition parameters from part 3 (which is saved as a pickle file after training).

To convert the parameters to their respective dictionaries, run the following (you can skip this step if you have done it in part 4):
```python3 part5.py```

The dictionary for emission parameters is saved as ```em_dic.p``` and the dictionary for transition parameters is saved as ```tr_dic.p```.

To output the 3rd best sequence, run the following:
```python3 part5.py```

The output file is saved as ```dev.p5.out``` in the EN folder.
