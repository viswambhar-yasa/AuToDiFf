# AuToDiFf
## Neural Network which performs calculate derivates and solve PDEs

Machine learning is used as advance curve fitting tool, i.e We fit a
curve amongst the known data points to know hitherto unknown data points
,only now using some sophisticated tools.There is also another characteristic
of Machine learning, which in my view is far less utilized than for curve-fit
applications and that is the ability to discover the underlying rules of the data
which is used to train. This project tries to utilize this aspect of machine
learning and deep learning, especially Neural Networks.

In this project, the models used are Long-Short term Memory Neural
Networks(LSTM) and the rules are given in the form of differential equations
with sufficient boundary conditions and therefore the output will be
the solution of the differential equation.

<p align="center">
<img src="https://github.com/viswambhar-yasa/AuToDiFf/blob/main/1_O73nlRM3-bWubvt6W-1YSg.png"/>
</p>


## How to install the Package?
1.Navigate to the folder containing the "setup.py" file
2.Execute the command "pip install ."

## How to run the test cases?
1.Navigate to the folder containing the files of test cases(named with suffix tests_)
2.Execute pytest \filename.py (eg. pytest tests_ops.py)

## External Packages Required apriori: Numpy and Matplotlib and sys 

This project is inspiried from "https://github.com/bgavran/autodiff" and few function are taken from that repo.
