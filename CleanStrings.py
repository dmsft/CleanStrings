import os
import sys
import time
import string
import random
import argparse
import signal

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# supporting modules
import nltk
import altair as alt
import pandas as pd
import numpy as np

# pretty output and progress bars
from rich import print
from rich.progress import track, open as rOpen
from rich.traceback import install
install(show_locals=True)

# classification labels
GOOD_LABEL = True
NOISE_LABEL = False

# ident pytorch device in use
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_printoptions(threshold=5)


# =================================================================================================
class LinesFeeder():
	"""Feed lines of `min_len` from `fname` or `text`, split at `max_len`, remove dupes and LFs."""

	def __init__(self, fname="", min_len=5, max_len=64, text="", chunk_past_max=True):

		if not fname and not text:
			raise ValueError("Need a filename or text.")

		self._inputFile = fname
		self._inputText = text
		self._minLen = min_len
		self._maxLen = max_len
		self._chunkPastMax = chunk_past_max


	def __iter__(self):

		if os.path.isfile(self._inputFile):
			return self.iter_file()
		elif len(self._inputText) >= self._minLen:
			return self.iter_text()
		else:
			raise ValueError("No input file or text.")


	# =============================================================================================
	def iter_file(self):

		with rOpen(self._inputFile, "r", encoding="utf8", refresh_per_second=5, description=self._inputFile) as fd:
			while True:
				lines = fd.readlines(256 * 1024)
				if not lines:
					break

				lines = self.parse_lines(lines)
				if len(lines) == 0:
					continue

				yield from lines


	# =============================================================================================
	def iter_text(self):

		lines = self._inputText.split("\n")
		lines = self.parse_lines(lines)

		if len(lines) == 0:
			return

		yield from lines


	# =============================================================================================
	def parse_lines(self, lines:list[str]):

		# remove newline characters
		lines = [ln.rstrip("\r\n") for ln in lines]
		all_lines = [ln for ln in lines if len(ln) <= self._maxLen]

		# split long lines
		if self._chunkPastMax:
			long_lines = (ln for ln in lines if len(ln) > self._maxLen)
			for line in long_lines:
				all_lines.extend(self.chunk_text(line))

		# exclude short lines
		lines = [ln for ln in all_lines if len(ln) >= self._minLen]

		# remove duplicates
		lines = list(set(lines))

		return lines


	# =============================================================================================
	def chunk_text(self, text:str):
		"""Split `text` into chunks of `max_len`."""

		lines = []
		for i in range(0, len(text), self._maxLen):
			chunk = text[i:i+self._maxLen]
			lines.append(chunk)

		return lines


# =================================================================================================
class NNDatasetBC(torch.utils.data.Dataset):

	def __init__(self, good_data, bad_data, max_len, pad_value=-1):

		self._pad = pad_value
		self._maxLen = max_len
		self._data = np.array([])
		self._labels = np.array([])

		# dataset counts
		self.GoodSize = len(good_data)
		self.BadSize = len(bad_data)
		self.TotSize = self.GoodSize + self.BadSize

		# tokenize and shuffle the data
		self.prep_data(good_data, bad_data)


	# =============================================================================================
	def __len__(self):
		return self.TotSize

	# =============================================================================================
	def __getitem__(self, idx):

		# Create a mask: 1 for actual data, 0 for padding
		data = self._data[idx]
		mask = (data != self._pad).astype(np.float32)

		return (data, mask, self._labels[idx])


	# =============================================================================================
	def prep_data(self, good_data, bad_data):

		# tokenize the data
		good_data = self.tokenize(good_data, self._maxLen, self._pad)

		# we don't always have the 2nd data set
		if len(bad_data) > 0:
			bad_data = self.tokenize(bad_data, self._maxLen, self._pad)
			data = np.concatenate((good_data, bad_data), axis=0)
		else:
			data = good_data

		# create labels
		labels = np.ones(len(good_data), dtype=np.float32)
		if len(bad_data) > 0:
			labels = np.concatenate((labels, np.zeros(len(bad_data), dtype=np.float32)), axis=0)

		# shuffle data and labels without loosing the order
		indices = np.arange(len(data))
		np.random.shuffle(indices)
		self._data = data[indices]
		self._labels = labels[indices]


	# =================================================================================================
	def tokenize(self, lines, max_len:int, pad_value=-1, divisor=127):
		"""Convert characters to normalized Ascii values and padded to `max_len`."""

		# run every char through ord() and pad to max_len
		arr = [list(map(ord, ln[:max_len])) + [pad_value*divisor]*(max_len-len(ln[:max_len])) for ln in lines]

		# convert to numpy and normalize (divide by max printable ASCII value)
		arr = np.array(arr, dtype=np.float32) / divisor

		return arr


# =================================================================================================
class BilinearModel(nn.Module):

	def __init__(self, input_size:int, hidden_size:int):
		super(BilinearModel, self).__init__()

		self._in_size = input_size
		self._hsize = hidden_size

		self._bil = nn.Bilinear(input_size, input_size, hidden_size, bias=False)
		self._linear_stack = nn.Sequential(
			nn.Linear(hidden_size, hidden_size//2),
			nn.ELU(),
			nn.Linear(hidden_size//2, hidden_size//3),
			nn.ELU(),
			nn.Linear(hidden_size//3, 1),
			nn.Sigmoid()
		)

		self._plot_idx = 0
		# self._viz = VizPlot(input_size*2, hidden_size)  # 0

	@property
	def input_dimensions(self):
		return self._in_size
	
	@property
	def hidden_dimensions(self):
		return self._hsize
	

	# =============================================================================================
	def forward(self, input:torch.Tensor, mask:torch.Tensor):

		# print(f"Input: {input.shape}\n{input}\nMask: {mask.shape}\n{mask}")

		# bilinear layer
		out = self._bil(input, mask)

		# run through the sequential layers
		out = self._linear_stack(out)

		#print(f"Out: {out.shape}\n{out}")
		return out


	# =============================================================================================
	def plot(self):

		weights = self._linear_stack[self._plot_idx].weight.detach().cpu().numpy()
		self._viz.plot_weights(weights)


	# =============================================================================================
	def save_graph(self):
		self._viz.save_graph(f"plot_bilinear_{self._plot_idx}.png")


# =================================================================================================
def load_corpus(name=""):
	"""Download corpus if not available."""

	try:
		nltk.data.find("corpora/" + name)
	except:
		try:
			nltk.download(name)
		except:
			return False

	return True


# =================================================================================================
def ClassifyNeuralNetwork(args):
	"""Read a text file and classify each line using the neural network model, printing good lines."""

	# extract required cli arguments
	threshold = args.threshold
	min_len = args.min_len
	model_file = args.model_file
	text_file = args.file
	workers = args.threads - 1
	debug = args.debug

	# load a saved model
	(model, dimensions) = LoadModel(model_file)
	model.eval()

	# load file contents, do not chunk and keep long lines
	lines = LinesFeeder(text_file, min_len, max_len=4096, chunk_past_max=False)
	lines = list(lines)  # generator to list

	# Prepare dataset and dataloader
	ds = NNDatasetBC(lines, [], dimensions)  # there's no noise data
	data_loader = torch.utils.data.DataLoader(
		ds, batch_size=256,
		shuffle=False, # retain order
		num_workers=workers,
		persistent_workers=False,
		pin_memory=True
	)

	results = []
	hits = 0
	with torch.no_grad():
		for idx, (inputs, masks, _) in enumerate(data_loader):
			prob = model(inputs.to(dev), masks.to(dev))

			# collect batches of results, in numpy format
			arr = prob.squeeze().cpu().numpy()
			results.extend(arr)

	# iterate all results and print the good ones
	for (idx, prob) in enumerate(results):
		if prob >= threshold:
			print(lines[idx])
			hits += 1
	
	# calc percentage shown vs total
	if debug:
		tot = len(lines)
		print(f"[b red]Shown {hits:,} out of {tot:,} [{hits/tot*100:.2f}%].")


# =================================================================================================
def SaveModel(file_prefix:str, model):
	"""Save the PyTorch `model` to file starting with `file_prefix`."""

	# filename is a combination of prefix, input dimensions, and hidden size
	fname = f"{file_prefix}_{model.input_dimensions}_{model.hidden_dimensions}.model"
	torch.save(model.state_dict(), fname)


# =================================================================================================
def LoadModel(file:str, input_size=0, hidden_size=0):
	"""Load a PyTorch model from `file`."""

	# filename is a combination of prefix, input dimensions, and hidden size

	# when sizes supplied use them as is
	if input_size and hidden_size:
		file = f"{file}_{input_size}_{hidden_size}.model"

	# if the file exists then it was requested via args
	# extract parameters from the filename
	elif os.path.isfile(file):
		parts = file.split("_")
		input_size = int(parts[1])
		hidden_size = int(parts[2].split(".")[0])
	else:
	# otherwise iterate CWD and pick the first file that matches the prefix
		files = os.listdir()
		prefix = file + "_"
		
		for fname in files:
			if not fname.startswith(prefix):
				continue

			parts = fname.split("_")
			input_size = int(parts[1])
			hidden_size = int(parts[2].split(".")[0])
			file = fname
			break
	
	# make sure we have the parameters
	if not input_size or not hidden_size:
		raise ValueError(f"Error loading model: {file}.")
	
	model = BilinearModel(input_size, hidden_size).to(dev)
	model.load_state_dict(torch.load(file, weights_only=True))

	return (model, input_size)


# =================================================================================================
def get_corpus_data(min_len:int, max_len:int) -> list[str]:
	"""Get unique lines from NLTK corpora, honor `min_len` and split at `max_len`."""

	# load the corpora, not critical if there's no data
	for name in ["names", "webtext", "gutenberg", "genesis"]:
		if not load_corpus(name):
			return []

	lines = []

	# names
	for fileid in nltk.corpus.names.fileids():
		lines += list(LinesFeeder(text="\n".join(nltk.corpus.names.words(fileid)), min_len=min_len, max_len=max_len))

	# webtext
	for fileid in nltk.corpus.webtext.fileids():
		full_text = nltk.corpus.webtext.raw(fileid)
		lines += list(LinesFeeder(text=full_text, min_len=min_len, max_len=max_len))

	# gutenberg
	for fileid in nltk.corpus.gutenberg.fileids():
		full_text = nltk.corpus.gutenberg.raw(fileid)
		lines += list(LinesFeeder(text=full_text, min_len=min_len, max_len=max_len))

	# genesis
	for fileid in nltk.corpus.genesis.fileids():
		full_text = nltk.corpus.genesis.raw(fileid)
		lines += list(LinesFeeder(text=full_text, min_len=min_len, max_len=max_len))

	return lines


# =================================================================================================
def make_random_strings(num:int, min_len:int, max_len:int) -> list[str]:
	"""Generate `num` random strings between `min_len` and `max_len` in length."""

	chars = string.ascii_letters + string.digits + string.punctuation
	rnd = random.randint
	choices = random.choices
	a = min_len
	b = max_len
	lines = ["".join(choices(chars, k=rnd(a, b))) for _ in track(range(num), "random")]

	return lines


# =================================================================================================
def get_data(good_file:str, noise_file:str, min_len:int, max_len:int):
	"""Get training data (user files, nltk, and random)."""

	good_data = []
	noise_data = []

	# NLTK corpora
	lines = get_corpus_data(min_len, max_len)
	good_data.extend(list(set(lines)))  # remove duplicates

	# good file
	lines = list(LinesFeeder(good_file, min_len, max_len))
	good_data.extend(list(set(lines)))  # remove duplicates

	# noise file
	lines = list(LinesFeeder(noise_file, min_len, max_len))
	noise_data.extend(lines)

	# random data, size is a percentage of the total data
	size = (len(good_data) + len(noise_data)) * 0.03
	lines = make_random_strings(int(size), min_len, max_len)
	noise_data.extend(lines)

	return (good_data, noise_data)


# =================================================================================================
def split_data(data:list[str], val_pct:float):

	size = int(len(data) * val_pct)
	train = data[size:]
	val = data[:size]

	return (train, val)


# =================================================================================================
def print_data_stats(good_set, noise_set):

	print(f"Training   Set (total = good + bad): ", end="")
	(good_size, bad_size) = (len(good_set[0]), len(noise_set[0]))
	tot_size = good_size + bad_size
	print(f"{tot_size:,} = {good_size:,} + {bad_size:,}", end="")
	print(f"  [{good_size/tot_size*100:.2f}% + {bad_size/tot_size*100:.2f}%]")

	print(f"Validation Set (total = good + bad): ", end="")
	(good_size, bad_size) = (len(good_set[1]), len(noise_set[1]))
	tot_size = good_size + bad_size
	print(f"{tot_size:,} = {good_size:,} + {bad_size:,}", end="")
	print(f"  [{good_size/tot_size*100:.2f}% + {bad_size/tot_size*100:.2f}%]")


# =================================================================================================
def save_loss_graph(losses:list[float]):
	"""Save the training loss graph as PNG image."""

	df_loss = pd.DataFrame(enumerate(losses), columns=["batch", "loss"])
	chart = alt.Chart(df_loss, height=1080, width=1920)
	chart = chart.mark_line().encode(alt.X("batch"), alt.Y("loss"))
	chart.save("training_loss.png")


# =================================================================================================
def test_predictions(model, dataloader, threshold=0.85):

	# stats counters
	hit_cnt = good_misses = bad_misses = 0
	all_probabilities = np.array([])
	all_labels = np.array([])

	model.to(dev).eval()
	with torch.no_grad():
		for (texts, masks, labels) in track(dataloader, "[b red]Final Validation"):
			# move tensor data to device
			texts = texts.to(dev)
			masks = masks.to(dev)

			# run the model on the batch
			output = model(texts, masks).squeeze().cpu().numpy()

			# collect the probabilities and labels
			all_probabilities = np.concatenate((all_probabilities, output), axis=0)
			all_labels = np.concatenate((all_labels, labels.numpy()), axis=0)

	# identify hits and misses
	expected_labels = (all_labels == 1)
	predicted_labels = (all_probabilities >= threshold)

	# count the hits and misses
	hit_cnt += np.sum(predicted_labels == expected_labels)
	good_misses += np.sum(predicted_labels & ~expected_labels)
	bad_misses += np.sum(~predicted_labels & expected_labels)

	# print the accuracy
	tot_size = len(dataloader.dataset)
	print(f"{hit_cnt:,} out of {tot_size:,} classified correctly [{hit_cnt/tot_size*100:.2f}%].")
	print(f"{good_misses:,} misclassified as {GOOD_LABEL} [{good_misses/hit_cnt*100:.2f}%].")
	print(f"{bad_misses:,} misclassified as {NOISE_LABEL} [{bad_misses/hit_cnt*100:.2f}%].")


# =================================================================================================
def calc_accuracy(predictions:torch.Tensor, ground_truth:torch.Tensor, threshold=0.85):
	"""Returns 0-1 accuracy for the given set of predictions and ground truth."""

	pred = predictions.squeeze()

	# must be 0.5 since round assumes that as the threshold
	temp = pred - (threshold - 0.5)
	rounded_predictions = torch.round(temp)

	# rounded_predictions = torch.floor(predictions + (1 - threshold))
	success = (rounded_predictions == ground_truth).float()  # convert bool to float for div
	accuracy = success.sum() / len(success)

	return accuracy.item()


# =================================================================================================
def run_batches(model, dataloader, lossFn, optimizer=None):

	loop_loss = loss_sum = accuracy_sum = 0
	loop_cnt = prev = items_cnt = 0

	batch_losses:list[float] = []
	epochs_losses = []
	tot_items = len(dataloader.dataset)

	start_time = time.time()
	for (texts, masks, labels) in dataloader:
		loop_cnt += 1
		items_cnt += len(texts)

		# move tensor data to device
		texts = texts.to(dev)
		masks = masks.to(dev)

		# model's forward pass
		outputs = model(texts, masks)

		# calculate the loss
		labels = labels.to(dev)
		loss = lossFn(outputs, labels.unsqueeze(1))
		loss_sum += loss.item()

		# accumulate accuracy for the batch
		accuracy_sum += calc_accuracy(outputs, labels)

		# no optimizer means  we're running validation, skip the rest
		if None == optimizer:
			continue

		# collect losses for later stats
		loop_loss += loss.item()
		batch_losses.append(loss.item())

		# backpropagate
		optimizer.zero_grad() # type: ignore
		loss.backward()
		optimizer.step() # type: ignore

		# periodically calculate the average loss of batches
		if loop_cnt % 100 == 0:
			epochs_losses.append(np.mean(batch_losses))
			# model.plot()  # plot the weights

		# progress update every N% of the total dataset
		if items_cnt // int(tot_items*0.13) > prev:
			elapsed = time.time() - start_time
			print(f"\tSamples {items_cnt:,} [{int(items_cnt/tot_items*100)}%]", end="")
			print(f"  Loss: {loop_loss/loop_cnt:.3f}", end="")
			print(f"  Speed: {int(items_cnt/elapsed):,} /sec")
			loop_loss = loop_cnt = 0
			prev = items_cnt // int(tot_items*0.13)

	# model.save_graph()
	return (loss_sum, accuracy_sum, epochs_losses)


# =================================================================================================
def train_batches(model, dataloader, lossFn, optim):

	model.train()
	return run_batches(model, dataloader, lossFn, optim)


# =================================================================================================
def validate(model, dataloader, lossFn):

	model.eval()
	with torch.no_grad():
		return run_batches(model, dataloader, lossFn)


# =================================================================================================
def train_epochs(model, train_loader, val_loader, lossFn, optimizer, numEpochs:int, model_file:str):

	print(f"Training on {len(train_loader.dataset):,} samples for {numEpochs} epochs ...")
	best_loss = float('inf')
	loss_sum = 0
	epoch_losses = []

	for epoch in range(numEpochs):
		_ = next(iter(train_loader))  # preemtively load the first batch, this triggeres workers spawning
		start = time.time()

		# train the model on the training set
		(loss, accuracy, losses) = train_batches(model, train_loader, lossFn, optimizer)
		loss /= len(train_loader)
		accuracy /= len(train_loader)
		print(f'[cyan]Training   Loss: {loss:.3f}  |  Accuracy: {accuracy*100:.2f}%')
		loss_sum += loss

		# graph the training loss
		epoch_losses.extend(losses)
		save_loss_graph(epoch_losses)

		# validate the model on the validation set
		(loss, accuracy, _) = validate(model, val_loader, lossFn)
		loss /= len(val_loader)
		accuracy /= len(val_loader)
		print(f'[cyan]Validation Loss: {loss:.3f}  |  Accuracy: {accuracy*100:.2f}%')

		# save the best model
		if loss < best_loss:
			SaveModel(model_file, model)
			best_loss = loss

		print(f"[cyan]Epoch: {epoch+1} | Avg Loss: {loss_sum/(epoch+1):.3f} | Elapsed: {time.time()-start:.2f} sec")


# =================================================================================================
def TrainNeuralNetwork(args):

	# show device: cuda or cpu
	print(f"[b red]Using {dev.type.upper()}")

	# extract required cli arguments
	good_file = args.file
	noise_file = args.noise_corpus
	min_len = args.min_len
	max_len = args.max_len
	threads = args.threads
	epochs = args.epochs
	hsize = args.hsize
	model_file = args.model_file
	predict_threshold = args.threshold

	# parameters
	learning_rate = 0.001
	hidden_size = hsize
	batches = 256
	workers = threads - 1
	val_pct = 0.10
	padding = -1

	# get and pre-parse the data
	(good_data, noise_data) = get_data(good_file, noise_file, min_len, max_len)

	# split the data into training and validation sets
	good_data = split_data(good_data, val_pct)
	noise_data = split_data(noise_data, val_pct)
	print_data_stats(good_data, noise_data)

	# create training loader
	train_ds = NNDatasetBC(good_data[0], noise_data[0], max_len, padding)  # train set
	train_loader = torch.utils.data.DataLoader(
		train_ds,
		batch_size=batches,
		shuffle=True,
		num_workers=workers,
		persistent_workers=True if workers > 0 else False,
		pin_memory=True
	)

	# create validation loader
	val_ds = NNDatasetBC(good_data[1], noise_data[1], max_len, padding)  # validation set
	val_loader = torch.utils.data.DataLoader(
		val_ds,
		batch_size=batches,
		shuffle=False,  # no need to shuffle while validating
		num_workers=1 if workers > 1 else 0,  # cap at 1 max
		persistent_workers=True if workers > 1 else False,
		pin_memory=True
	)

	# create the model, loss function and optimizer
	model = BilinearModel(max_len, hidden_size).to(dev)
	loss_fn = nn.BCELoss()
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	# optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# train the model (also saves the one with lowest validation loss)
	train_epochs(model, train_loader, val_loader, loss_fn, optimizer, epochs, model_file)

	# reload the model for testing
	(model, _) = LoadModel(model_file, max_len, hidden_size)

	# run a validation test on the final model
	test_predictions(model, val_loader, predict_threshold)


# =================================================================================================
if __name__ == "__main__":
	pr = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		description="Train and classify 'clean' strings.")
	pr.add_argument("file", help="Filename with strings (when training used as `clean` set).")
	pr.add_argument("-l", "--min_len", type=int, default=5, help="Minimum line length.")
	pr.add_argument("-y", "--threshold", type=float, default=0.85, help="Classification threshold.")
	pr.add_argument("-m", "--model_file", help="Model filename prefix.", default="CleanStrings")
	pr.add_argument("-t", "--train", action="store_true", help="Train a classifier.")
	pr.add_argument("-z", "--noise_corpus", help="Filename with `bad` strings (train only).")
	pr.add_argument("-j", "--threads", type=int, default=1, help="Number of threads (train only).")
	pr.add_argument("-d", "--debug", action="store_true", help="Show classification probabilities.")
	pr.add_argument("--max_len", type=int, default=32, help="Maximum line length (train only).")
	pr.add_argument("--epochs", type=int, default=5, help="Number of epochs (train only).")
	pr.add_argument("--hsize", type=int, default=30, help="Hidden layer size (train only).")
	args = pr.parse_args()
	start_time = time.time()

	# handle Ctrl-C
	signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

	if args.train:
		TrainNeuralNetwork(args)
	else:
		ClassifyNeuralNetwork(args)

	if args.debug:
		print(f"[b red]Elapsed time: {(time.time()-start_time)/60:.2f} min.", file=sys.stderr)
