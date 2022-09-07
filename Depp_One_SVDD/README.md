This is an implementation of [Deep One SVDD paper](https://proceedings.mlr.press/v80/ruff18a.html). The results are mostly simialar to original implementation.

# Results:
Average results over 10 runs
## Mnist:
| implementation | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
|----------------|------|------|------|------|------|------|------|------|------|------|
| original       | 98.0 | 99.7 | 91.7 | 91.9 | 94.9 | 88.5 | 98.3 | 94.6 | 93.9 | 96.5 |
| my             | 97.6 | 99.4 | 88.1 | 90.8 | 93.4 | 86.5 | 97.5 | 93.9 | 92.8 | 96.1 |

## cifar10:
| implementation | AIRPLANE | AUTOMOBILE | BIRD | CAT  | DEER | DOG  | FROG | HORSE | SHIP | TRUCK |
|----------------|----------|------------|------|------|------|------|------|-------|------|-------|
| original       | 61.7     | 65.9       | 50.8 | 59.1 | 60.9 | 65.7 | 67.7 | 67.3  | 75.9 | 73.1  |
| my             | 61.7     | 59.1       | 49.3 | 57.5 | 55.2 | 62.8 | 55.5 | 58.3  | 75.7 | 65.1  |