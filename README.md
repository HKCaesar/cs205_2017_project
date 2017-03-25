# CS205 Project
CS205, Spring 2017
Computing Foundations for Computational Science
M. Manjunathaiah (Manju)

```
This repository was created in partial fulfillment of the requirements of CS 205 in Spring 2017.
```

======
Kareem Carr, Eric Dunipace, Charlotte Lloyd
======

BACKGROUND:
------

The consumption of commercial sexual services is not uncommon: 15-18% of men living in the U.S. report having at least one paid sexual encounter since they were 18 years old (General Social Survey, 1991-2006), and roughly 1% have visited a sex worker in the past year (Monto 1999; Monto and McRee 2005). Over the past 15 years, technological innovation combined with tougher policies against street prostitution have catalyzed the growth of markets for Internet-based sexual service providers. Scholars have investigated the concomitant rise of the “girlfriend experience” (GFE), which includes cuddling, conversation, and even dinner dates in addition to the provision of more traditional sexual services. However, very little is known about the structure of this doubly clandestine market, which is both socially stigmatized and illegal.

DATA: We will use a unique data from the world’s largest sex work review website (www.theeroticreview.com) covering the period from April 1998 to July 2011 and including 584,513 reviews written by 118,683 reviewers regarding encounters with 113,703 sex workers. Crucially, extensive work has already been undertaken with a customized dictionary to code for the presence of the six common sexual acts: (1) kissing, (2) massage, (3) hand job,(4) fellatio, (5) cunninlingus, and (6) vaginal intercourse. When compared to human coding of sexual acts in the reviews, the dictionary classifies 92% of acts correctly and falsely codes 11% of acts. 

METHODOLOGY:  

We wish to understand the underlying types of clients and providers. It is our belief that clients and providers are drawn from an underlying distribution of types. In order to identify these types, we propose using K-means clustering. The K-means problem produces a partitional clustering of the observed data with the smallest within-cluster variance, which can be interpreted as a residual sum of squares. The within-cluster variance is sensitive to the choice of the number of clusters K. Several solutions have been proposed to address this limitation. One of the most successful is the Gap statistic. Another approach suggested by Broderick et al is incorporating a penalty function. We will explore various approaches to determining K which will involve subsampling the data and iteratively incorporating distribution of estimates derived from subsamples.
