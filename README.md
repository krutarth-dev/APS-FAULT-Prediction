# APS-FAULT-Prediction
## 1.1 Project Detail
The dataset consists of data collected from heavy scania trucks in everyday usage. The system Focus is APS which generates pressurized air that is utilized in various functions in a truck, Such as braking and gear changes. The datasets positive class corresponds to component failures for a specific component of the APS system. The negative class corresponds to trucks with failures for components not related to the APS System.

## 1.2 Project Profile
The Training set contains 60000 examples in total in which 59000 belong to negative class and 1000 belong to positive class. The test set contains 16000 examples. There are 171 attributes per record. The dataset used for this project was imported from UCL ML Repository.

## 1.3 Project Definition
Our goal in this project is to predict if the truck needs to be serviced or not and minimize the cost associated with:
1) Unnecessary Checks done by mechanic which means negative labeled point is classified
as positive.
2) Missing a faulty truck which may cause a breakdown. If positive labeled point is classified as negative.

## 2 Data Dictionary
1) The attribute names of the data have been anonymized for proprietary reasons. It consists of both single numerical counters and histograms consisting of bins with different conditions.
2) Typically the histograms have open-ended conditions at each
end. For example if we measuring the ambient temperature "T" then the histogram could be defined with 4 bins where: bin 1 collect values for temperature T < -20 bin collect values for temperature T >= -20 and T < 0 bin 3 collect values for temperature T >= 0 and T < 20 bin 4 collect values for temperature T > 20
 Cost1=10andCost2=500
The total cost of a prediction model the sum of Cost_1 multiplied by the number of Instanceswithtype1failureandCost_2 withthenumberofinstanceswithtype2 failure,resultinginaTotal_cost.InthiscaseCost_1 referstothecostthatan unnessecary check needs to be done by an mechanic at an workshop, while Cost_2 refer to the cost of missing a faulty truck, which may cause a breakdown. Total_cost = Cost_1 * No_Instances + Cost_2 * No_Instances.

## 3.1 Problem Encountered and Possible Solutions
1) Need to Handle many Null values in almost all columns
2) No low-latency requirement.
3) Interpretability is not important.
4) misclassification leads the unecessary repair costs.

## 3.2 Limitation and Future Work
The limitations of ML model are:
1) The target classes are highly imbalanced
2) Class imbalance is a scenario that arises when we have unequal distribution of class in a dataset i.e. the
no. of data points in the negative class (majority class) very large compared to that of the positive class
(minority class)
3) If the imbalanced data is not treated beforehand, then this will degrade the performance of the classifier
model.
4) Hence we should handle imbalanced data with certain methods.

### The future work of our ML model:
The APS Fault detection model should be published on any deployment services like ASW EC2 where it can be updated in real time and can be viewed by users in a more convenient manner
