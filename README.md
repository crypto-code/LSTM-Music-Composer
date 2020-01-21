# LSTM-Music-Composer
Use an LSTM to Generate Beethoven Music 


## What is an LSTM ?
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used. LSTMs are explicitly designed to avoid the vanishing gradient problem. 

<p align="center">
<img src="https://github.com/crypto-code/LSTM-Music-Composer/blob/master/assets/model.png" align="middle" />  </p>

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

For more info check out this [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## Usage: 

* First collect the midi files to train from, a few samples of beethoven's [midi files](https://github.com/crypto-code/LSTM-Music-Composer/tree/master/beeth) are already provided.

* To train the lstm run on the collected training data, run train.py

```
python train.py --help
Using TensorFlow backend.
usage: train.py [-h] --input INPUT --name NAME [--epoch EPOCH]

LSTM Music Generator

optional arguments:
  -h, --help     show this help message and exit
  --input INPUT  Directory containing input music samples. eg: ./data
  --name NAME    Name of the Music
  --epoch EPOCH  Number of training epochs
  ```
  
* To generate new music samples run generate.py
```
python generate.py --help
Using TensorFlow backend.
usage: generate.py [-h] --name NAME --output OUTPUT

LSTM Music Generator

optional arguments:
  -h, --help       show this help message and exit
  --name NAME      Name of the Music
  --output OUTPUT  Output file name
```

# G00D LUCK

For doubts email me at:
atinsaki@gmail.com
