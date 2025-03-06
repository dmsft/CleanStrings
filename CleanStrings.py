import os
import sys
import time
import string
import pickle
import random
import argparse
import signal
import multiprocessing as mp

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(threshold=5)

# supporting modules
import nltk
import altair as alt
import pandas as pd
import numpy as np
import win32api, win32process, win32con

# pretty output and progress bars
from rich import print as rPrint
from rich.progress import track, open as rOpen
from rich.traceback import install
install(show_locals=True)

# classification labels
GOOD_LABEL = True
NOISE_LABEL = False

# pytorch device in use, will be configured in the NN model class
dev = None


# =============================================================================================
def LowerPriority():
	"""Set the priority class of this process (normal, above, or below)."""

	# https://mhammond.github.io/pywin32/modules.html
	prcl = win32process.BELOW_NORMAL_PRIORITY_CLASS
	pid = win32api.GetCurrentProcessId()
	handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
	win32process.SetPriorityClass(handle, prcl)
	win32api.CloseHandle(handle)


# =================================================================================================
class LinesFeeder():
	"""Feed lines of `min_len` from `fname` or `text`, split at `max_len`, remove dupes and LFs."""

	def __init__(self, fname="", min_len=5, max_len=64, text="", chunk_past_max=True, verbose=False):

		if not fname and not text:
			raise ValueError("Need a filename or text.")

		self._inputFile = fname
		self._inputText = text
		self._minLen = min_len
		self._maxLen = max_len
		self._chunkPastMax = chunk_past_max
		self._verbose = verbose


	def __iter__(self):

		if os.path.isfile(self._inputFile):
			return self.iter_file()
		elif len(self._inputText) >= self._minLen:
			return self.iter_text()
		else:
			raise ValueError("No input file or text.")


	# =============================================================================================
	def iter_file(self):

		if self._verbose:
			ctx = rOpen(self._inputFile, "r", encoding="utf8", refresh_per_second=5, description=self._inputFile)
		else:
			ctx = open(self._inputFile, "r", encoding="utf8")

		with ctx  as fd:
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

		# remove duplicates (keep order)
		lines = list(dict.fromkeys(lines))

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

	def __init__(self, good_data, bad_data, max_len, pad_value=-1, shuffle=True):

		self._pad = pad_value
		self._maxLen = max_len
		self._shuffle = shuffle
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
		if self._shuffle:
			indices = np.arange(len(data))
			np.random.shuffle(indices)
			self._data = data[indices]
			self._labels = labels[indices]
		else:
			self._data = data
			self._labels = labels


	# =================================================================================================
	def tokenize(self, lines, max_len:int, pad_value=-1, divisor=127):
		"""Convert characters to normalized Ascii values and padded to `max_len`."""

		# run every char through ord() and pad to max_len
		arr = [list(map(ord, ln[:max_len])) + [pad_value*divisor]*(max_len-len(ln[:max_len])) for ln in lines]

		# convert to numpy and normalize (divide by max printable ASCII value)
		arr = np.array(arr, dtype=np.float32) / divisor

		return arr


# =================================================================================================
class BaseModel(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


	# =================================================================================================
	@staticmethod
	def SaveLossGraph(losses:list[float]):
		"""Save the training loss graph as PNG image."""

		df_loss = pd.DataFrame(enumerate(losses), columns=["batch", "loss"])
		chart = alt.Chart(df_loss, height=1080, width=1920)
		chart = chart.mark_line().encode(alt.X("batch"), alt.Y("loss"))
		chart.save("training_loss.png")


	# =================================================================================================
	@staticmethod
	def CalcAccuracy(predictions:torch.Tensor, ground_truth:torch.Tensor, threshold=0.85):
		"""Returns 0-1 accuracy for the given set of predictions and ground truth."""

		# print(f"Predictions: {predictions.shape} {predictions}")
		# print(f"Ground Truth: {ground_truth}")

		pred = predictions.squeeze()
		# print(f"Pred: {pred}")

		# must be 0.5 since round assumes 0.5 as the threshold
		temp = pred - (threshold - 0.5)
		# print(f"Temporary: {temp}")

		rounded_predictions = torch.round(temp)
		# print(f"Rounded Predictions: {rounded_predictions}")

		# rounded_predictions = torch.floor(predictions + (1 - threshold))
		success = (rounded_predictions == ground_truth).float()  # convert bool to float for div
		accuracy = success.sum() / len(success)

		# print(f"Success: {success}")
		# print(f"Accuracy: {accuracy} {accuracy.item()}")
		# exit()

		return accuracy.item()


	# =================================================================================================
	@staticmethod
	def Validate(model, dataloader, lossFn):

		model.eval()
		with torch.no_grad():
			return model.run_batches(model, dataloader, lossFn)


	# =================================================================================================
	@staticmethod
	def Train(model, train_loader, val_loader, lossFn, optimizer, numEpochs:int, model_file:str):

		print(f"Training on {len(train_loader.dataset):,} samples for {numEpochs} epochs ...")

		best_loss = float('inf')
		loss_sum = 0
		epoch_losses = []

		for epoch in range(numEpochs):
			_ = next(iter(train_loader))  # preemtively load the first batch, this triggeres workers spawning
			start = time.time()

			# train the model on the training set
			(loss, accuracy, losses) = model.train_batches(model, train_loader, lossFn, optimizer)
			loss /= len(train_loader)
			accuracy /= len(train_loader)
			print(f'[cyan]Training   Loss: {loss:.3f}  |  Accuracy: {accuracy*100:.2f}%')
			loss_sum += loss

			# graph the training loss
			epoch_losses.extend(losses)
			model.SaveLossGraph(epoch_losses)

			# validate the model on the validation set
			(loss, accuracy, _) = model.Validate(model, val_loader, lossFn)
			loss /= len(val_loader)
			accuracy /= len(val_loader)
			print(f'[cyan]Validation Loss: {loss:.3f}  |  Accuracy: {accuracy*100:.2f}%')

			# save the best model
			if loss < best_loss:
				model.Save(model_file)
				best_loss = loss

			print(f"[cyan]Epoch: {epoch+1} | Avg Loss: {loss_sum/(epoch+1):.3f} | Elapsed: {time.time()-start:.2f} sec")


	# =================================================================================================
	@staticmethod
	def train_batches(model, dataloader, lossFn, optim):

		model.train()
		return model.run_batches(model, dataloader, lossFn, optim)


	# =================================================================================================
	@staticmethod
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
			accuracy_sum += model.CalcAccuracy(outputs, labels)

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

			# progress update every N% of the total dataset
			if items_cnt // int(tot_items*0.13) > prev:
				elapsed = time.time() - start_time
				print(f"\tSamples {items_cnt:,} [{int(items_cnt/tot_items*100)}%]", end="")
				print(f"  Loss: {loop_loss/loop_cnt:.3f}", end="")
				print(f"  Speed: {int(items_cnt/elapsed):,} /sec")
				loop_loss = loop_cnt = 0
				prev = items_cnt // int(tot_items*0.13)

		return (loss_sum, accuracy_sum, epochs_losses)


# =================================================================================================
class LinearModel(nn.Module):

	def __init__(self, input_size:int, hidden_size:int):
		super(LinearModel, self).__init__()

		self._in_size = input_size
		self._hsize = hidden_size

		self._linear_stack = nn.Sequential(
			nn.Linear(input_size*2, hidden_size),  # double because of mask
			nn.ELU(),
			nn.Linear(hidden_size, hidden_size//2),
			nn.ELU(),
			nn.Linear(hidden_size//2, hidden_size//3),
			nn.ELU(),
			nn.Linear(hidden_size//3, 1),
			nn.Sigmoid()
		)


	@property
	def input_dimensions(self):
		return self._in_size

	@property
	def hidden_dimensions(self):
		return self._hsize


	# =============================================================================================
	def forward(self, input:torch.Tensor, mask:torch.Tensor):


		# concat input with the mask
		cat = torch.cat((input, mask), dim=1)  # Shape: (batch_size, input_size + mask_size)

		# run through the sequential layers
		out = self._linear_stack(cat)

		return out


# =================================================================================================
class BilinearModel(BaseModel):

	def __init__(self, input_size:int, hidden_size:int):
		super().__init__()

		global dev
		dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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


	@property
	def input_dimensions(self):
		return self._in_size

	@property
	def hidden_dimensions(self):
		return self._hsize


	# =============================================================================================
	def forward(self, input:torch.Tensor, mask:torch.Tensor):

		# bilinear layer
		out = self._bil(input, mask)

		# run through the sequential layers
		out = self._linear_stack(out)

		return out


	# =================================================================================================
	def TestPredictions(self, dataloader, threshold=0.85):

		# stats counters
		hit_cnt = good_misses = bad_misses = 0
		all_probabilities = np.array([])
		all_labels = np.array([])

		self.to(dev).eval()
		with torch.no_grad():
			for (texts, masks, labels) in track(dataloader, "[b red]Final Validation"):
				# move tensor data to device
				texts = texts.to(dev)
				masks = masks.to(dev)

				# run the model on the batch
				output = self.forward(texts, masks).squeeze().cpu().numpy()

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
	def Save(self, file_prefix:str):
		"""Save the PyTorch `model` to file starting with `file_prefix`."""

		# filename is a combination of prefix, input dimensions, and hidden size
		fname = f"{file_prefix}_{self.input_dimensions}_{self.hidden_dimensions}.model"
		torch.save(self.state_dict(), fname)


	# =================================================================================================
	@classmethod
	def Load(cls, file:str, input_size=0, hidden_size=0):
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
class NaiveBayesClassifier():

	def __init__(self, model_file="", verbose=False):

		self._verbose = verbose

		if model_file:
			self._classifier = self.load(model_file)


	# =============================================================================================
	def Save(self, fname:str):

		with open(fname, "wb") as fd:
			pickle.dump(self._classifier, fd)

		if self._verbose:
			print(f"[b red]Model saved to {fname}.", file=sys.stderr)


	# =============================================================================================
	def Train(self, good_data=[], noise_data=[]):

		if self._verbose:
			print(f"[b red]Labeling data ...", file=sys.stderr)

		train_set = self.vectorize(good_data[0], noise_data[0])
		val_set = self.vectorize(good_data[1], noise_data[1], False)

		if self._verbose:
			start = time.time()
			print(f"[b red]Training Naive Bayes classifier ...", file=sys.stderr)

		self._classifier = nltk.NaiveBayesClassifier.train(train_set)
		print(f"Accuracy: {nltk.classify.accuracy(self._classifier, val_set):.2f}")

		if self._verbose:
			self._classifier.show_most_informative_features()
			print(f"[b red]Training took {time.time()-start:.2f} sec.", file=sys.stderr)


	# =============================================================================================
	def Classify(self, lines:list[str]) -> list[tuple[str, float]]:

		out = []
		for line in lines:
			features = NaiveBayesClassifier.extract_features(line)
			probability = self._classifier.prob_classify(features)

			good_prob = probability.prob(GOOD_LABEL)
			# noise_prob = probability.prob(NOISE_LABEL)

			out.append((line, good_prob))

		return out


	# =============================================================================================
	def load(self, fname:str):

		if self._verbose:
			print(f"[b red]Loading model from {fname} ...", file=sys.stderr)

		try:
			with open(fname, "rb") as fd:
				cl = pickle.load(fd)
				return cl
		except:
			raise ValueError("Error loading classifier.")


	# =============================================================================================
	@staticmethod
	def extract_features(line:str) -> dict:
		"""Extract features."""

		# remove punctuation, digits and whitespace
		# table = {ch: " " for ch in string.punctuation + string.digits + string.whitespace}
		# line = text.translate(str.maketrans(table))
		vowels = set("aeiou")
		ft = {}

		tot_len = len(line)
		ft["tot_len"] = tot_len

		# count words and their average length
		arr = line.split(" ")
		a = len(arr)
		b = sum(map(len, arr))
		ft["num_words"] = a
		ft["avg_word_len"] = int(b / a)

		ft["num_vowels"] = sum([1 for c in line if c in vowels])
		ft["num_digits"] = sum([1 for c in line if c.isdigit()])
		ft["num_spaces"] = sum([1 for c in line if c.isspace()])
		ft["num_punct"] = sum([1 for c in line if c in string.punctuation])
		ft["num_upper"] = sum([1 for c in line if c.isupper()])
		ft["num_lower"] = sum([1 for c in line if c.islower()])

		# sum and average of char values
		a = sum([ord(c) for c in line])
		ft["chr_sum"] = a
		ft["chr_avg"] = int(a / tot_len)

		# english word frequency
		# ft["freq"] = int(utils.CalcEnglishFreq(line))

		return ft


	# ===============================================================================================
	def vectorize(self, good_data, noise_data, shuffle=True):
		"""Convert text to feature vectors and apply labels.."""

		start = time.time()
		# apply labels, make a new tuple list for each good/noise set
		# good = [(extract_features(line), GOOD_LABEL) for line in track(good)]
		# noise = [(extract_features(line), NOISE_LABEL) for line in track(noise)]

		# good  = nltk.classify.apply_features(extract_features, good, False)
		# good = [(ft, GOOD_LABEL) for ft in good]
		# noise = nltk.classify.apply_features(extract_features, noise, False)
		# noise = [(ft, NOISE_LABEL) for ft in noise]

		# apply labels
		good = [(line, GOOD_LABEL) for line in good_data]
		noise = [(line, NOISE_LABEL) for line in noise_data]

		# combine the two sets
		labeled_data = good + noise

		if shuffle:
			for _ in range(3):
				random.shuffle(labeled_data)

		labeled_set  = nltk.classify.apply_features(NaiveBayesClassifier.extract_features, labeled_data, True)

		if self._verbose:
			print(f"[b red]Labeled {len(labeled_data):,} items in {time.time()-start:.2f} sec.", file=sys.stderr)

		return labeled_set


# =================================================================================================
def TrainNaiveBayes(args):

	# extract required cli arguments
	good_file = args.file
	noise_file = args.noise_corpus
	min_len = args.min_len
	max_len = args.max_len
	model_file = args.model_file + ".pickle"
	verbose = args.debug
	val_pct = 0.10

	# get and pre-parse the data
	(good_data, noise_data) = get_data(good_file, noise_file, min_len, max_len, verbose)

	# split the data into training and validation sets
	good_data = split_data(good_data, val_pct)
	noise_data = split_data(noise_data, val_pct)
	print_data_stats(good_data, noise_data)

	# train the classifier
	nb = NaiveBayesClassifier("", verbose)
	nb.Train(good_data, noise_data)

	# save the classifier
	nb.Save(model_file)


# =================================================================================================
cls_nn_model = None
cls_nn_dimensions = 0
def ClassifyNeuralNetwork(model_prefix="", lines=[]) -> list[tuple[str, float]]:
	"""Read a text file and classify each line using the neural network model, printing good lines."""

	global cls_nn_model, cls_nn_dimensions

	# load a saved model
	if not cls_nn_model:
		(model, dimensions) = BilinearModel.Load(model_prefix)
		model.eval()
		cls_nn_model = model
		cls_nn_dimensions = dimensions
	else:
		model = cls_nn_model
		dimensions = cls_nn_dimensions

	# Prepare dataset and dataloader
	ds = NNDatasetBC(lines, [], dimensions, shuffle=False)  # there's no noise data
	data_loader = torch.utils.data.DataLoader(
		ds, batch_size=256,
		shuffle=False, # retain order (this is important)
		num_workers=0,
		pin_memory=True
	)

	results = []
	with torch.no_grad():
		for i, (inputs, masks, _) in enumerate(data_loader):
			prob = model(inputs.to(dev), masks.to(dev))

			# collect batches of results, in numpy format
			arr = prob.squeeze().cpu().numpy()
			results.extend(arr)

	# iterate all results
	out = []
	for (i, prob) in enumerate(results):
		out.append((lines[i], prob))

	# print(f"[b red]Shown {hits:,} out of {tot:,} [{hits/tot*100:.2f}%].", file=sys.stderr)
	return out


# =================================================================================================
def ClassifyMain(args):

	# extract required cli arguments
	algo = args.algo
	file = args.file
	min_len = args.min_len
	max_len = args.max_len
	verbose = args.debug
	model_file = args.model_file
	threshold = args.threshold
	threads = args.threads

	# init queues
	input_queue = mp.Queue(threads * 3)
	results_queue = mp.Queue(threads * 3)
	procs = []

	# spawn the dispatcher
	t = mp.Process(target=classify_dispatch, args=(input_queue, file, min_len, max_len, threads, verbose))
	t.start()
	procs.append(t)

	# spawn workers
	for _ in range(threads):
		t = mp.Process(target=classify_worker, args=(input_queue, results_queue, algo, model_file, verbose))
		t.start()
		procs.append(t)

	# iterate results queue
	num_threads_done = 0
	cnt = total = 0
	while True:
		results = results_queue.get()

		# when the worker thread is done it sends None
		if results is None:
			num_threads_done += 1

			# we need to make sure all threads are finished before breaking the loop
			if num_threads_done == threads:
				break
			else:
				continue

		cnt += print_classification(results, threshold, verbose)
		total += len(results)

	# Wait for all threads to finish
	for t in procs:
		t.join()

	rPrint(f"[b red]Shown {cnt:,} out of {total:,} [{cnt/total*100:.2f}%].", file=sys.stderr)


# =================================================================================================
def classify_dispatch(queue:mp.Queue, file:str, min_len:int, max_len:int, threads:int, verbose=False):
	"""Dispatch lines to worker processes."""

	# load the data
	lines = LinesFeeder(file, min_len, max_len, chunk_past_max=False, verbose=verbose)
	lines = list(dict.fromkeys(lines))  # generator to uniq list (order preserved)

	chunk_size = len(lines) // threads
	for i in range(0, len(lines), chunk_size):
		queue.put(lines[i:i+chunk_size])

	# signal the workers to stop
	for _ in range(threads):
		queue.put(None)


# =================================================================================================
def classify_worker(in_queue:mp.Queue, out_queue:mp.Queue, algo:str, model_file:str, verbose=False):
	"""Worker process to classify lines."""

	# load NB model
	if algo in ["nb", "both"]:
		nb = NaiveBayesClassifier(model_file + ".pickle")

	while True:
		lines = in_queue.get()
		if not lines:
			break

		if algo == "nb":
			results = nb.Classify(lines)
		elif algo == "nn":
			results = ClassifyNeuralNetwork(model_file, lines)
		elif algo == "both":
			nb_results = nb.Classify(lines)
			nn_results = ClassifyNeuralNetwork(model_file, lines)

			assert len(nb_results) == len(nn_results)
			assert nb_results[0][0] == nn_results[0][0]

			results = []
			for i, (line, nb_prob) in enumerate(nb_results):
				(_, nn_prob) = nn_results[i]
				avg_prob = (nb_prob + nn_prob) / 2
				results.append((line, avg_prob))

				if verbose:
					try:
						rPrint(f"{line[:32]:<37}\t[{nb_prob:.3f}\t{nn_prob:.3f}] = {avg_prob:.3f}")
					except:
						print(f"{line[:32]:<37}\t[{nb_prob:.3f}\t{nn_prob:.3f}] = {avg_prob:.3f}")

		out_queue.put(results)

	# signal to the consumer that we're done
	out_queue.put(None)


# =================================================================================================
def print_classification(results=[], threshold=0.85, verbose=False) -> int:
	"""Print classification results."""

	cnt = 0
	for (line, prob) in results:
		if verbose:
			hit_color = "[green]" if prob >= threshold else "[red]"
			rPrint(f"{line[:32]:<37}\t{hit_color}{prob:.3f}")
			cnt += 1
			continue

		if prob < threshold:
			continue

		print(line)
		cnt += 1

	return cnt


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
	lines = ["".join(choices(chars, k=rnd(a, b))) for _ in range(num)]

	return lines


# =================================================================================================
def get_data(good_file:str, noise_file:str, min_len:int, max_len:int, verbose=False):
	"""Get training data (user files, nltk, and random)."""

	good_data = []
	noise_data = []

	# NLTK corpora
	lines = get_corpus_data(min_len, max_len)
	good_data.extend(list(set(lines)))  # remove duplicates

	# good file
	lines = list(LinesFeeder(good_file, min_len, max_len, verbose=verbose))
	good_data.extend(list(set(lines)))  # remove duplicates

	# noise file
	lines = list(LinesFeeder(noise_file, min_len, max_len, verbose=verbose))
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
def TrainNeuralNetwork(args):

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
	verbose = args.debug

	# parameters
	learning_rate = 0.001
	hidden_size = hsize
	batches = 256
	workers = threads - 1
	val_pct = 0.10
	padding = -1

	# get and pre-parse the data
	(good_data, noise_data) = get_data(good_file, noise_file, min_len, max_len, verbose)

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
	# model = LinearModel(max_len, hidden_size).to(dev)
	model = BilinearModel(max_len, hidden_size).to(dev)
	loss_fn = nn.BCELoss()
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	# optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# show device: cuda or cpu
	print(f"[b red]Using {dev.type.upper()}") # type: ignore

	# train the model (also saves the one with lowest validation loss)
	model.Train(model, train_loader, val_loader, loss_fn, optimizer, epochs, model_file)

	# reload the model for testing
	(model, _) = BilinearModel.Load(model_file, max_len, hidden_size)

	# run a validation test on the final model
	model.TestPredictions(val_loader, predict_threshold)


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
	pr.add_argument("-j", "--threads", type=int, default=0, help="Number of threads.")
	pr.add_argument("-d", "--debug", action="store_true", help="Show classification probabilities.")
	pr.add_argument("--algo", choices=["nb", "nn", "both"], default="both", help="Algorithm to use.")
	pr.add_argument("--max_len", type=int, default=32, help="Maximum line length (train only).")
	pr.add_argument("--epochs", type=int, default=5, help="Number of epochs (train only).")
	pr.add_argument("--hsize", type=int, default=30, help="Hidden layer size (train only).")
	args = pr.parse_args()
	start_time = time.time()

	# handle Ctrl-C
	signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

	# show CUDA status
	if args.debug:
		if torch.cuda.is_available():
			rPrint(f"[cyan]CUDA enabled.", file=sys.stderr)
		else:
			rPrint(f"[red]CUDA disabled.", file=sys.stderr)

	if args.train:
		if args.threads == 0:
			args.threads = 5

		if args.algo == "nb":
			TrainNaiveBayes(args)
		elif args.algo == "nn":
			TrainNeuralNetwork(args)
		else:
			print(f"[e] Unknown algorithm: {args.algo}.", file=sys.stderr)
	else:
		if args.threads == 0:
			args.threads = os.cpu_count()
			LowerPriority()

		ClassifyMain(args)

	if args.debug:
		rPrint(f"[b red]Elapsed time: {(time.time()-start_time)/60:.2f} min.", file=sys.stderr)
