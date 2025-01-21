# CleanStrings

Train a binary classifier using a simple neural network and a probabilistic one to distinguish between *clean* and *noisy* strings. The intention is to cleanup **strings** output for faster triage in malware analysis and forensics. Currently, the approach is to use a Multilayer Perceptron (MLP) network with just a handful of layers and a Naive Bayes classifier. The text gets fed through both and probability results are averaged before the prediction threshold.


# Install

The following requirements can be installed using `pip`.

```bash
py -m pip install --upgrade --user nltk altair pandas numpy rich
py -m pip install --upgrade --user torch --index-url https://download.pytorch.org/whl/cu124
```


# Usage

Use the supplied model files which were trained on 21 million samples with 70/30 split for *clean* vs *noise*. Given this simple design it achieves 96% accuracy.

```bash
py CleanStrings.py -h

usage: CleanStrings.py [-h] [-l MIN_LEN] [-y THRESHOLD] [-m MODEL_FILE] [-t] [-z NOISE_CORPUS] [-j THREADS] [-d]
                       [--algo {nb,nn}] [--max_len MAX_LEN] [--epochs EPOCHS] [--hsize HSIZE]
                       file

Train and classify 'clean' strings.

positional arguments:
  file                  Filename with strings (when training used as `clean` set).

options:
  -h, --help            show this help message and exit
  -l MIN_LEN, --min_len MIN_LEN
                        Minimum line length. (default: 5)
  -y THRESHOLD, --threshold THRESHOLD
                        Classification threshold. (default: 0.85)
  -m MODEL_FILE, --model_file MODEL_FILE
                        Model filename prefix. (default: CleanStrings)
  -t, --train           Train a classifier. (default: False)
  -z NOISE_CORPUS, --noise_corpus NOISE_CORPUS
                        Filename with `bad` strings (train only). (default: None)
  -j THREADS, --threads THREADS
                        Number of threads (train only). (default: 1)
  -d, --debug           Show classification probabilities. (default: False)
  --algo {nb,nn}        Algorithm to use. (default: nn)
  --max_len MAX_LEN     Maximum line length (train only). (default: 32)
  --epochs EPOCHS       Number of epochs (train only). (default: 5)
  --hsize HSIZE         Hidden layer size (train only). (default: 30)
```

Given the output from **strings**, or any other text file, it will only output lines based on default threshold of **0.85** probability of being *clean*. For a more strict *cleanup* use `-y` to increase threshold.

```bash
py CleanStrings.py -d -y 0.95 memdump.txt
...
Files/Go/src/runtime/histogram.go
*bbolt.txPending
runtime.traceLocker.GoEnd
commitFreelist@4@
runtime.mapKeyError
golang.org/x/text/message/catalog.ErrNotFound
*mapstruct
...

Shown 1,987 out of 2,006 [99.05%].
```


# Train

To train your own model you will need two corpus text files, one with *clean* data and the other *noisy*. Experiment with `max_len`, `epochs`, and `hsize` parameters to achieve desired results. The `threads` helps when corpus files are large e.g. 5 threads on 21 million samples was optimal. Also, on top of using the corpus text files we'll use some English corpora from **NLTK** and a small random strings sampling.

```bash
py CleanStrings.py -t -z corpus_noise.txt corpus_clean.txt --epochs 5 --max_len 12 -j 5 --hsize 150

Training   Set (total = good + bad): 21,016,423 = 14,669,964 + 6,346,459  [69.80% + 30.20%]
Validation Set (total = good + bad): 2,335,157 = 1,629,995 + 705,162  [69.80% + 30.20%]
Training on 21,016,423 samples for 3 epochs ...
        Samples 2,732,288 [13%]  Loss: 0.172  Speed: 13,995 /sec
        Samples 5,464,320 [26%]  Loss: 0.137  Speed: 13,925 /sec
        Samples 8,196,608 [39%]  Loss: 0.126  Speed: 13,840 /sec
        Samples 10,928,640 [52%]  Loss: 0.120  Speed: 13,884 /sec
        Samples 13,660,672 [64%]  Loss: 0.115  Speed: 13,826 /sec
        Samples 16,392,960 [78%]  Loss: 0.113  Speed: 13,857 /sec
        Samples 19,124,992 [91%]  Loss: 0.110  Speed: 13,800 /sec
Training   Loss: 0.126  |  Accuracy: 93.17%
Validation Loss: 0.095  |  Accuracy: 95.63%
Epoch: 1 | Avg Loss: 0.126 | Elapsed: 1617.83 sec
...
Final Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:52
2,250,592 out of 2,335,157 classified correctly [96.38%].
18,176 misclassified as True [0.81%].
66,389 misclassified as False [2.95%].
```
