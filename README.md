# Text Color Predictor
A simple feed forward net written from scratch in Python to help me improve my understanding of neural nets and explore the effects of tuning various parameters.

Given a backgound color - predicts whether a black or white text should be displayed for best readability. 

You can quickly create your own data set using this [tool](tools/data-sample-generator/data_sample_generator.html). Drop it in `data/` and update the trainer to read from your data instead.
## Run

```
$ python3 train.py
```