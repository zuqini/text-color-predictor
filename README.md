# Text Color Predictor
A simple feed forward neural net written from scratch in Python for educational purposes.

Given the RGB values of a solid background color - predicts whether a black or white text should be displayed for best readability.

The current iteration has an accuracy of 97~% based on my own test set.

<img width="715" alt="screen shot 2019-01-02 at 2 15 44 am" src="https://user-images.githubusercontent.com/5790854/50582974-72493780-0e34-11e9-9829-a5f7d98f5db1.png">

## Running

```
$ python3 train.py
```
View your test set results in the generated `results.html`.

You can also create your own data set for training/testing using this [tool](tools/data-sample-generator/data_sample_generator.html). Drop the generated json in the `data/` folder and update the trainer to read from that instead.
