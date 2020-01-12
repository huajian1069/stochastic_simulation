Stochastic simulation project


The generated random number data is stored in data directory. Because the class parameter information used to generate data 
is also stored, the file is serialized into pickle file, which may not work in another computer. The first time running the program, you may need to regenerate 
data, after that you can plot the result multiple times without generating it again.

The plot figures are stored in figures directory.

The points of project are responded explicitly at Project_ex2/3/4.py file.
Some commonly used function are stored in Project_utils.py file.

Notice: Before plotting the result, you need to generate data or load data from disk, otherwise the program will report '
ParallelTempering class no attribute of self.xs' exception.