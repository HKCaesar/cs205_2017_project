# CS205 Project: Using Parallelized Classification to Determine the Structure of a Clandestine Market

CS205, Spring 2017
Computing Foundations for Computational Science
M. Manjunathaiah (Manju)

------
Kareem Carr, Eric Dunipace, Charlotte Lloyd
------

<p align="center">
<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/writeup/KmeansAnimation.gif" width="450">
</p>

```
This repository was created in partial fulfillment of the requirements of CS 205 in Spring 2017.
```

## Substantive Problem

> <i>We wish to understand the underlying types of clients of sex work, which we believe are drawn from an underlying distribution of types. In order to identify these types, we propose using K-means clustering.</i>

The consumption of commercial sexual services is not uncommon: 15-18% of men living in the U.S. report having at least one paid sexual encounter since they were 18 years old (General Social Survey, 1991-2006), and roughly 1% have visited a sex worker in the past year (Monto 1999; Monto and McRee 2005). Over the past 15 years, technological innovation combined with tougher policies against street prostitution have catalyzed the growth of markets for Internet-based sexual service providers. Scholars have investigated the concomitant rise of the “girlfriend experience” (GFE), which includes cuddling, conversation, and even dinner dates in addition to the provision of more traditional sexual services. However, very little is known about the structure of this doubly clandestine market, which is both socially stigmatized and illegal.

We will use a unique data from the world’s largest sex work review website (www.theeroticreview.com) covering the period from April 1998 to July 2011 and including 584,513 reviews written by 118,683 reviewers regarding encounters with 113,703 sex workers. Crucially, extensive work has already been undertaken with a customized dictionary to code for the presence of the six common sexual acts: (1) kissing, (2) massage, (3) hand job,(4) fellatio, (5) cunninlingus, and (6) vaginal intercourse. When compared to human coding of sexual acts in the reviews, the dictionary classifies 92% of acts correctly and falsely codes 11% of acts. 

## Methodology: K-means

> <i>K-means is a popular method of clustering that iteratively calculates group means and reassigns observations to clusters until convergence is achieved.</i>

Invented in 1955, K-means is a simple clustering method that remains one of the most popular and widely used algorithms today (Jain 2010). The K-means problem produces a partitional clustering of the observed data with the smallest within-cluster variance, which can be interpreted as a residual sum of squares. The within-cluster variance is sensitive to the choice of the number of clusters K. Several solutions have been proposed to address this limitation. One of the most successful is the Gap statistic. Another approach suggested by Broderick et al is incorporating a penalty function. We will explore various approaches to determining K which will involve subsampling the data and iteratively incorporating distribution of estimates derived from subsamples.

Given the popularity of this clustering method, it is unsurprising that there has been corresponding interest in parallel implementations of K-means (Farivar et al. 2008; Hong-tao et al. 2009; Kanungo et al. 2002; Li et al. 2010; Shalom, Dash, and Tue 2008; Wu and Hong 2011; Zechner and Granitzer 2009; Zhao, Ma, and He 2009). [do we need to commment on this existing literature?]

[should we mention existing pyCUDA implementations?]
https://github.com/shackenberg/cukmeans.py
https://github.com/dbelll/kmeans
https://github.com/serban/kmeans
https://bitbucket.org/malthejorgensen/kmeans-gpu-nbi

## Parallel Architecture

> <i>summary</i>

<img align="left" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/writeup/arch-cpus.png"  width="400">

In this crowded field, our contribution is to create a flexible K-means implementation using mpi4py + pyCUDA that will be accessible and useful to data scientist, many of whom are most comfortable programming in Python rather than in traditional languages of parallel processing including Fortran, C, and C++. Since data scientists may be deeply unfamiliar with parallel hardware, we paid special attention to designing a flexible architecture suitable for any configuration involving at least one CUDA-ready GPU. 

In an ideal hardware configuration, we would expect a host CPU to control <i>c</i> CPUs that each in turn control <i>g</i> GPUs. 

<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/writeup/arch-ideal.png">

1. First, the host CPU (rank 0 in the MPI framework) would partition the data into <i>c</i> subsets.  
2. Second, each CPU would communicate with its <i>g</i> GPUs to handle the arithmetically intense aspect of the K-means calculations, i.e. the computation of distance and reassigning of labels. 
3. Third, the host CPU (rank 0 in the MPI framework) would re-assemble the results of the subsets into the final means, labels, and distortion score. 

However, due to the limitations of Odyssey's hardware configuration, we were not able to implement our ideal parallel architecture described above. Instead.... 

<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/writeup/arch-odyssey.png">

## Performance

> <i>We have yet to complete a hybrid mpi4py + pyCUDA implemenation due to hardware difficulties on Odyssey. Currently, our pyCUDA implementation is fastest. Both our mpi4py and pyCUDA implementations are faster than the sequential K-means algorithm written in Python. However, the "stock" K-means written purely in C is still fastest up to n=100,000.</i>

<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/plots/time-bar-1000-6-5.png">
<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/plots/time-bar-10000-6-5.png">
<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/plots/time-bar-50000-6-5.png">
<img align="center" src="https://raw.githubusercontent.com/kareemcarr/cs205_2017_project/master/analysis/plots/time-bar-100000-6-5.png">

## Substantive Findings

> <i>summary</i>

The K-means clustering of consumption patterns revealed three major groups: ... 

## Works Cited

<sub>Broderick, T., Kulis B., and Jordan M. (2013), MAD-Bayes: MAP-based Asymptotic Derivations from Bayes. https://arxiv.org/abs/1212.2126 </sub>

<sub>Farivar, R., Rebolledo, D., Chan, E., & Campbell, R. H. (2008). A Parallel Implementation of K-Means Clustering on GPUs. In PDPTA (Vol. 13, pp. 212–312). Retrieved from https://pdfs.semanticscholar.org/0638/dc0565cb11191ab1e2b91cd19b630cfa8c34.pdf </sub>

<sub>Hong-tao, B., Li-li, H., Dan-tong, O., Zhan-shan, L., & He, L. (2009). K-Means on Commodity GPUs with CUDA. In 2009 WRI World Congress on Computer Science and Information Engineering (Vol. 3, pp. 651–655). https://doi.org/10.1109/CSIE.2009.491 </sub>

<sub>Jain, A. K. (2010). Data clustering: 50 years beyond K-means. Pattern Recognition Letters, 31(8), 651–666. https://doi.org/10.1016/j.patrec.2009.09.011 </sub>

<sub>Kanungo, T., Mount, D. M., Netanyahu, N. S., Piatko, C. D., Silverman, R., & Wu, A. Y. (2002). An efficient k-means clustering algorithm: analysis and implementation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 881–892. https://doi.org/10.1109/TPAMI.2002.1017616 </sub>

<sub>Li, Y., Zhao, K., Chu, X., & Liu, J. (2010). Speeding up K-Means Algorithm by GPUs. In 2010 10th IEEE International Conference on Computer and Information Technology (pp. 115–122). https://doi.org/10.1109/CIT.2010.60 </sub>

<sub>MacQueen, J. Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics, 281--297, University of California Press, Berkeley, Calif., 1967. http://projecteuclid.org/euclid.bsmsp/1200512992 </sub>

<sub>Monto, Martin A. 1999. Clients of Street Prostitutes in Portland, Oregon, San Francisco and Santa Clara, California, and Las Vegas, Nevada, 1996-1999: Version 1. Retrieved May 9, 2014 http://www.icpsr.umich.edu/NACJD/studies/02859/version/1</sub>

<sub>Monto, Martin A. and Nick McRee. 2005. “A Comparison of the Male Customers of Female Street Prostitutes With National Samples of Men.” International Journal of Offender Therapy and Comparative Criminology 49(5):505–29. https://www.ncbi.nlm.nih.gov/pubmed/16260480</sub>

<sub>Shalom, S. A. A., Dash, M., & Tue, M. (2008). Efficient K-Means Clustering Using Accelerated Graphics Processors. In SpringerLink (pp. 166–175). Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-85836-2_16 </sub>

<sub>Tibshirani, R., Walther, G. and Hastie, T. (2001), Estimating the number of clusters in a data set via the gap statistic. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63: 411–423. (http://onlinelibrary.wiley.com/doi/10.1111/1467-9868.00293/abstract) </sub>

<sub>Wu, J., & Hong, B. (2011). An Efficient k-Means Algorithm on CUDA. In 2011 IEEE International Symposium on Parallel and Distributed Processing Workshops and Phd Forum (pp. 1740–1749). https://doi.org/10.1109/IPDPS.2011.331 </sub>

<sub>Zechner, M., & Granitzer, M. (2009). Accelerating K-Means on the Graphics Processor via CUDA. In 2009 First International Conference on Intensive Applications and Services (pp. 7–15). https://doi.org/10.1109/INTENSIVE.2009.19 </sub>

<sub>Zhao, W., Ma, H., & He, Q. (2009). Parallel K-Means Clustering Based on MapReduce. In SpringerLink (pp. 674–679). Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-10665-1_71 </sub>
