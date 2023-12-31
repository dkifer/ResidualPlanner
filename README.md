# ResidualPlanner

Sourse code for the paper An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions.

## Description

ResidualPlanner is a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly run out of memory, even for relatively small datasets).

The algorithms are implemented in the following files.
-  **class_resplan.py**:  A unified framework for selection, measurement and reconstruction. Support calculation for sum of variance and max variance.
-  **selection_only/class_sumvar.py**: Custimized for faster selection for sum of variance. 
-  **selection_only/class_maxvar.py**: Custimized for faster selection for max variance.


## Usage

The following code shows how to use ResidualPlanner on a synthetic dataset. 

```python
from class_resplan import ResidualPlanner

# specify domains size and column name for each attribute in the dataset
domains = [10] * 5
col_names = ['0', '1', '2', '3', '4']
# privacy budget for rho-zcdp
rho = 1
system = ResidualPlanner(domains)

# choose marginals, (0, ) represents an 1-way marginal on the first attribute
# (3, 4) represents a 2-way marginal on the 4th and 5th attribute.
attributes = [(0, ), (1, ), (2, ), (3, 4), (0, 1, 2), (1, 4), (2, 3)]
for att in attributes:
  system.input_mech(att)

# input the dataset, here we use a synthetic dataset
data = pd.DataFrame(np.zeros([10_000, 5]), columns=col_names)
system.input_data(data, col_names)

# select, measure and reconstruct
# choice="sumvar" ==> optimize sum of variance
# choice="maxvar" ==> optimize max variance
sum_var = system.selection(choice="sumvar", pcost=2*rho)
system.measurement()
system.reconstruction()
```


## Datasets
There are two dataset in the experiment.
- **adult.csv**: the Adult dataset has 14 attributes, each having domain sizes $100,100,100,99,85,42,16,15,9,7,6,5,2,2$, respectively, resulting in a record domain size of $ 6.41 * 10^{17}$.
- **Synth-$n^d$**: Here $d$ refers to the number of attributes (we experiment from $d=2$ to $d=100$) and $n$ is the domain size of each attribute. The running times of the algorithms only depend on $n$ and $d$ and not on the records in the synthetic data.


## Run Experiment

The following files contain codes for experiments in the paper. Note that due to randomness and different machine performance, the result may not be exactly the same as the numbers in the paper. 
- **exp_synthetic_dataset.py:** Time and loss performance on synthetic datasets.
- **exp_large_dataset.py:** Time and loss performance on large datasets.
- **exp_reconstruct** Time performance for reconstruction step.

