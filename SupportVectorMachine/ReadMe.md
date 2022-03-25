# Support Vector Machine Study

#### SupportVectorMachine.py and SupportVectorMachineCVXOPT.py
* SupportVectorMachine.py is a SVM classifier from scratch. And SupportVectorMachineCVXOPT.py contains SVM classifiers uses CVXOPY library for optimization. 
* The summary and difference of the two implementations are listed in the table below. 

<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160161326-21cb1984-3d74-45e2-acbe-87208ffda3d1.png"  width="644" height="289">
</p></pre>

* Results
  * SupportVectorMachine.py 
    * linear 
  * SupportVectorMachineCVXOPT.py
    * linear, nonlinear, soft 

* Some discussions 
  *  When using CVXOPT encounter some errors
     *  ValueError: Rank(A) < p or Rank([P; A; G]) < n
     *  Domain error
     *  Max out the iteration numbers error
  * Reduce the number of training points from 100 to less than 20 helps resolve the ValueError: Rank(A) < p or Rank([P; A; G]) < n error. But then  encounter domain error and max out the iteration numbers error. 
  * Removing the Ax = b constraint helps resolve the domain error and max out the iteration numbers error.
  * Tried different solver options like reltol, abstol, and kktsolver but didn't help much. 
  * Tried out CVXPY but could not install it. [This discussion](https://github.com/robertmartin8/PyPortfolioOpt/issues/396 "Google's Homepage") mentioned to install an older version of VS v14 2017.  
  * Changed the data set for nonlinear problem to three cluster for ease of visualization. Also changed polynomial solver from p=3 to p=2. 


#### KaggleBreastCancerSVM.py and KaggleBreastCancerSVM.ipynb 
* Both scripts uses scikit-learn library for SVM classification.  
* KaggleBreastCancerSVM.ipynb tried out some techniques
  *  Data set normalization using using RobustScaler() 
  *  Correlation analysis between parameters and heatmap visualization
  *  Box plot, violin plot, and swarm plot using Seaborn library  
  *  Parameter feature reduction 
  *  Confussion matrcies visualization 
  *  Tried out different classifiers: KNN, SVM, Logistic Regression, XGboost 
* Results 
  * 


