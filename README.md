# cds_language
My repository for the course of Language Analytics


## Assignmnet 5: (Un)supervised machine learning
I was not able to make the code run in the terminal which is a shame, and if you have any advice it is greatly appreciated. It is however possible to run the notebook code "assignment5.py" in the src/ folder. Here the visualization of the LDA model can be seen, but I could not figure out how to get these visualizations saved through the terminal. 
I think it might be possible to see the output however, if you clone this repository and look at the html in "output/" on the jupyter notebook. 

For this assignment I worked with Johanne Brandhøj Würtz, Emil Buus Thomsen & Christoffer Kramer. 
We wanted to examine how an LDA model would categorize words within the specific online community of the reddit thread r/wallstreetsbets, as there is a certain internally understood jargon. (Dataset: https://www.kaggle.com/unanimad/reddit-rwallstreetbets)

This code does however just take a sample of 10000 in the dataset that contains over 1 mio. titles to examine and to run this entire set it takes a long time. After having run a code of different samples the lowest coherence I got was 0.49 which is pretty low, while the highest was 0.6 meaning that it does give some strange results. When looking at the LDA results it was also a bit difficult to understand as to how it categorized words and made the topics. I choose to sort by three categories as it seemed like that was the best way to display the differences. (Anything above this would be topics clustered on top of another topic) 