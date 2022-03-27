# Support Vector Machine Study

### SupportVectorMachine.py and SupportVectorMachineCVXOPT.py
* SupportVectorMachine.py is a SVM classifier from scratch. 
* SupportVectorMachineCVXOPT.py contains SVM classifiers uses CVXOPY library for optimization. 
* The summary and difference of the two implementations are listed in the table below. 

<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160176273-87bd3b31-abb4-4825-a759-8bdf6c95e20e.png"  width="550">
</p></pre>

#### Results
* SupportVectorMachine.py 
  * Figure below shows the result of the simple linear SVM classifier. The circle dots are training data and star dots are testing data. 

<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160172551-1b38ac15-d3f8-48f5-b12c-624c848508b8.png"  width="362">
</p></pre>

* SupportVectorMachineCVXOPT.py
  * Below shows the training data and hyperplane of linear kernel, polynomial kernel, gaussian kernel, and soft margin. 
<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160293643-f899d5ab-6cce-4657-905d-60f91827a105.png"  width="1000"> </br>
<img src="https://user-images.githubusercontent.com/86133411/160293532-e18099ed-243e-4d95-b4bd-492df587e00e.png"  width="640">
</p></pre>

* Some discussions 
  *  When using CVXOPT encounter some errors
     *  ValueError: Rank(A) < p or Rank([P; A; G]) < n
     *  Domain error
     *  Max out the iteration numbers error
  * Reduce the number of training points from 100 to less than 20 helps resolve the ValueError: Rank(A) < p or Rank([P; A; G]) < n error. But then  encounter domain error and max out the iteration numbers error. 
  * Removing the Ax = b constraint helps resolve the domain error and max out the iteration numbers error.
  * Tried different solver options like reltol, abstol, and kktsolver but didn't help much. 
  * Tried out CVXPY but could not install it. [This discussion](https://github.com/robertmartin8/PyPortfolioOpt/issues/396 "Google's Homepage") mentioned to install an older version of VS v14 2017. 
    *  Later successfully install CVXPY with python 3.6. But encounter DCP rule error when data points become large. 
  * Changed the data set for nonlinear problem to three cluster for ease of visualization. Also changed polynomial solver from p=3 to p=2. 


### KaggleBreastCancerSVM.py and KaggleBreastCancerSVM.ipynb 
* Both scripts uses scikit-learn library.  
* KaggleBreastCancerSVM.ipynb tried out some techniques. 
  *  Tried out different classifiers: KNN, random forest, SVM, logistic regression, XGboost 
  *  Data set normalization using using RobustScaler() 
  *  Correlation analysis between parameters and heatmap visualization
  *  Box plot, violin plot, and swarm plot using Seaborn library  
  *  Parameter feature reduction 
  *  Confussion matrcies visualization 

* Confusion matrix of KNN (left), Random Forest (middle), and SVM (right)
<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160294113-4f504b06-8660-4127-a6e6-5dccec8d40e3.png"  width="300"> <img src="https://user-images.githubusercontent.com/86133411/160294107-5b36fbdd-cdd1-473c-8575-d8a6b5915121.png"  width="300"> <img src="https://user-images.githubusercontent.com/86133411/160304088-775be4df-312f-4af5-b917-25ecbf083ccc.png"  width="300"> 
</p></pre>
* Confusion matrix of Logistic Regression (left) and XGBoost (right)
<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160294105-2578cac9-eb63-426d-9369-3f817ccfa764.png"  width="300"> <img src="https://user-images.githubusercontent.com/86133411/160294099-9764cc3f-719e-484e-a9d3-584afe81956f.png"  width="300"> 
</p></pre>

* Heatmap and joint plot. Shows correlations between parameters. 
<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160294117-4d5c947f-346c-4214-9b5b-6d5781614e07.png"  width="500"> </br>
<img src="https://user-images.githubusercontent.com/86133411/160294129-61ae2888-e966-47b4-98d1-3b82bd05a47b.png"  width="300"> <img src="https://user-images.githubusercontent.com/86133411/160294125-8325cdd1-9622-403c-ba61-d5792289fbdd.png"  width="300">
</p></pre>

* Box plot, swarm plot, and violin plot. Shows parameters' statistics.  
<pre><p align="center">
<img src="https://user-images.githubusercontent.com/86133411/160294122-592f83dc-c162-49d4-8be8-e0b5c78cefb7.png"  width="500"> </br>
<img src="https://user-images.githubusercontent.com/86133411/160294119-fd847368-0599-433c-a7bf-2ef17ccf3fc4.png"  width="500"> </br>
<img src="https://user-images.githubusercontent.com/86133411/160294121-b4152eb3-b5f3-4edf-8042-4127e3669faf.png"  width="500"> </br>
</p></pre>
