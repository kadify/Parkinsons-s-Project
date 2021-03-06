# Detecting Parkinson's using Multidimensional Voice Program Analysis parameters

Multidimensional Voice Program Analysis (MDVP) is an advanced system which allows for the measurement of 33 quantitative voice parameters. 195 MDVP analysis data records were used to build a model capable of detecting vocal attributes possibly linked with tremors associated with Parkinson's Disease. From these 195 recordings, supplied by 31 people. 23 had Parkinson's and 8 people did not. Of the 33 parameters MDVP extrapolates, 22 were supplied in the dataset.

## Question
*Is it possible to detect whether someone has Parkinson's Disease from MDVP parameters?*

## Hypothesis

> *H<sub>0</sub>: μ<sub>1</sub> =  μ<sub>2</sub>*<br>
>*H<sub>0</sub>:* There is no discernible difference between the speech frequencies, vocal jitter, or other parameters between a person with Parkinson's Disease and one that does not. Therefore, it will not be possible to create a model to predict if someone has Parkinson's.


　



>*H<sub>1</sub>: μ<sub>1</sub> ≠  μ<sub>2</sub>*<br>
> *H<sub>1</sub>:* There is a difference between the speech frequencies and parameters between people with Parkinson's disease and those that do not and therefore, a model will be able to predict, with better accuracy than random guessing, whether or not someone has Parkinson's Disease.

## Exploratory Data Analysis & Data Cleaning

#### Class Imbalance 
![](images/class_imbalance2.png)

There was an observed class imbalance due to there being data logs from 22 people with Parkinson's but only 8 from people who did not have Parkinson's. Data sampling methods had to be used to handle this. 

#### Feature Collinearity
![](images/initital_heatmap.png)

Looking at the correlation between each of the 22 MDVP parameters there was a fairly decent collinearity (similarity between features)  with status (whether the recording was from someone with Parkinson's or from someone without Parkinson's). The highest correlations between the parameters and status being spread1 and PPE with a positive 0.56 and 0.53 correlation, respectively. PPE had a positive correlation 0.53. MDVP.Fo(Hz), MDVP.Flo(Hz), and HNR had negative correlations of -0.38, -0.38 and -0.36. These were the parameters I looked at first when modeling, however with most of the other parameters having at least a 0.15 correlation, I initially thought it may be difficult to fit with only 5 parameters when all the others are also correlated. 

Additionally, looking at the correlations, the following parameters have correlations with each other above 0.90 and therefore are not likely contributing much additional information due to them being **colinear**:

**MDVP:Jitter(%)**
- MDVP:Jitter(Abs) & MDVP:Jitter(%) (0.94)
- MDVP:RAP & MDVP:Jitter(%) (**0.99**)
- MDVP:PPQ & MDVP:Jitter(%) (0.97)
- Jitter:DDP & MDVP:Jitter(%) (**0.99**)
- NHR & MDVP:Jitter(%) (0.91)

**MDVP:Jitter(Abs)**
- MDVP:RAP & MDVP:Jitter(Abs) (0.92)
- MDVP:PPQ & MDVP:Jitter(Abs) (0.90)
- Jitter:DDP & MDVP:Jitter(Abs) (0.92)

**MDVP:RAP**
- MDVP:PPQ & MDVP:RAP (0.96)
- Jitter:DDP & MDVP:RAP (**1.0**)
- NHR & MDVP:RAP (0.92)

**Jitter:DDP**
- MDVP:PPQ & Jitter:DDP (0.96)
- NHR & Jitter:DDP (0.92)

**MDVP:RAP**
- Jitter:DDP & MDVP:PPQ (0.96)

**MDVP:Shimmer**
- MDVP:Shimmer(dB) & MDVP:Shimmer (**0.99**)
- Shimmer:APQ3 & MDVP:Shimmer (**0.99**)
- Shimmer:APQ5 & MDVP:Shimmer (**0.98**)
- MDVP:APQ & MDVP:Shimmer (0.95)
- Shimmer:DDA & MDVP:Shimmer (**0.99**)

**MDVP:Shimmer(dB)**
- Shimmer:APQ3 & MDVP:Shimmer(dB) (0.96)
- Shimmer:APQ5 & MDVP:Shimmer(dB) (0.96)
- MDVP:APQ & MDVP:Shimmer(dB) (0.97)
- Shimmer:DDA & MDVP:Shimmer(dB) (0.96)

**Shimmer:APQ3**
- Shimmer:APQ5 & Shimmer:APQ3 (0.96)
- MDVP:APQ & Shimmer:APQ3 (0.90)
- Shimmer:DDA & Shimmer:APQ3 (**1.0**)

**Shimmer:APQ5**
- MDVP:APQ & Shimmer:APQ5 (0.95)
- Shimmer:DDA & Shimmer:APQ5 (0.96)

**MDVP:APQ**
- Shimmer:DDA & MDVP:APQ (0.90)

**Spread1**
- PPE & Spread1 (0.96)

During exploration, many of the parameters were colinear (having a correlation value of 1.0) and I decided to drop the following parameters:
1. Shimmer:APQ3
2. Shimmer:APQ5
3. MDVP:RAP
4. MDVP:Shimmer(dB)

![](images/second_heatmap.png)

From the secondary correlation map, it was clear there were still a lot of parameters that had significant collinearity with each other. This is most likely because the parameters are transformations of other parameters, therefore I manually selected 3 parameters to investigate for my model:
1. MDVP: Fo (Hz)
2. MDVP: Flo (Hz)
3. spread1

![](images/my_heatmap.png)

---

![](images/my_scatter.png)
I created a scattermatrix to visualize the distributions between the parameters and the target (status). Since status is classified into 0s (healthy people) and 1s (people with Parkinson's), I chose that I would need to experiment with models better suited to classifying data such as:
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier


## Model Methodology
This was an exploratory analysis in the best method to create a model that limits false negative results. Recall was used to determine the fidelity the model represented in correctly detecting if someone did indeed have Parkinson's Disease.

 ![](images/Recall.png)

As a result, precision, depending on number of incorrect false positive results, was not investigated. With regards to why recall was investigated and not precision, in medicine, minimizing the number of diagnoses missed (false negatives) is imperative while number of preemeptive diagnoses (false positives) is not generally as costly.

Multiple types of models were sampled and used to determine which resulted in the highest recall for the test data (data never before seen by the model) in conjunction with the highest accuracy (indicating that there isn't a tradeoff of less false negatives for more false positives). It was found that a Gradient Boosted Classifying model with an exponential loss was the most appropriate and resulted in the fewest missed diagnoses.

## Model
After performing a gridsearch of the mentioned models above, a gradient boosted classifying model returned the best prediction with the lowest number of missed diagnoses (recall). 

|![](images/naive_gbc.png)|![](images/optimized_gbc.png)|
|-|-|
|A naive gradient boosted classifier model on the manually selected features. Using the MDVP: Fo(Hz), MDVP:Flo(Hz), and spread1 features, a precision of 0.896, recall of 0.909, accuracy of 0.852, and a ROC Area Under the Curve value of 0.795 represents the ability of the model to generalize well to unseen data. | An optimized gradient boosted classifier model on the manually selected features. Using the MDVP: Fo(Hz), MDVP:Flo(Hz), and spread1 features, a precision of 0.899, recall of 0.939, accuracy of 0.875, and a ROC Area Under the Curve value of 0.811 represents the ability of the model to generalize well to unseen data.|

The optimized gradient boosted model generalizes better than the naive model and represents a higher recall score over the naive model. The optimized model performs better resulting in a 93.9% recall score which means out of 100 people with parkinson's, this model would be able to correctly diagnose ~94 of them. 


## Formal Presentation

![](images/Parkinsons_presentation.png)


## Continued work
If I were to continue work with this model I would like to see what few records the model isn't able to correctly diagnose as parkinson's and see what the respective MDVP: Fo(Hz), MDVP:Flo(Hz), and spread1 feature values are. This model technically isn't predicting whether someone has parkinson's or not. The model, on a mechanical basis, is just predicting the probability that someone has parkinson's based on the mentioned feature values. It is possible that the diagnoses that the model misses are because the scores for these respective people are not in the range of other people with parkinson's. This could be a result of them being at an earlier stage of parkinson's and therefore do not have significant vocal changes and therefore still sound 'normal'. It would be interesting to explore whether that is the case or if the model is truly just misdiagnosing cases of parkinson's that otherwise fall into the normal range other people with parkinson's demonstrate for these features.
