import random
from copy import deepcopy
from collections import defaultdict, Counter

start = "<s>"
class NGramModel:
    def __init__(self, token_list, n=1):
        """
        n gram model, based on Bayes theory
        :param token_list: a list of tokenized text
        :param n: the number of tokens considered when making the calculation
        """
        self._ngrams = defaultdict(lambda: 0)    # key: tuple of ngram, value: the count of that particular ngram
        self._n_1grams = defaultdict(lambda: 0)
        self._n = n
        self._pred_distr = None    # key: n-1 gram, value: dict (key: nth word, value: prob)
        self.token_list = token_list
        self.update_model(token_list)

    @property
    def token_num(self):
        """
        :return: the total count of ngrams
        """
        sum = 0
        for count in self._ngrams.values():
            sum += count
        return sum

    @property
    def vocab_size(self, unk = False):
        """
        return the vocabulary size of the traning set
        :param unk: is ther any unk tokens in the training set?
        :return: vocab size (include unk)
        """
        return len(set(self.token_list)) if unk else len(set(self.token_list)) + 1

    def __repr__(self):
        return "%i-gram model with %i unique keys" % (self._n, len(self))

    def __len__(self):
        """Returns the number of unique keys in the model"""
        return len(self._ngrams.keys())

    def update_model(self, token_list):
        """
        method for updating ngram model
        update the dict of ngrams
        reset prediction distribution
        :param token_list: a list of tokenized text
        :return: None
        """
        token_seq = []
        for token in token_list:
            token_seq.append(token)
            if len(token_seq) > self._n:
                token_seq.pop(0)
            if len(token_seq) == self._n:
                self._ngrams[tuple(token_seq)] += 1
                self._n_1grams[tuple(token_seq[:-1])] +=1
        self._pred_distr = None

    def predict(self, given=None):
        """
        give a 1~n-1 gram, return next token
        if no given gram, return the most likely ngram
        first initiate the dict prediction distribution
        then make predictions based on it
        :param given: 0~n-1 gram, should be a tuple
        :return: a token
        """
        if self._pred_distr is None:
            self._pred_distr = defaultdict(lambda: defaultdict(lambda: 0))
            before_n_counter = Counter([ngram[:-1] for ngram in self._ngrams])
            for ngram in self._ngrams:
                before_n = ngram[:-1]
                nth = ngram[-1]
                self._pred_distr[before_n][nth] += self._ngrams[ngram]
            for before_n in self._pred_distr:
                for nth in self._pred_distr[before_n]:
                    self._pred_distr[before_n][nth] /= before_n_counter[before_n]

        if given is None:
            population = list(self._ngrams.keys())
            return random.choices(population, self.vectorize(population))[0]
        else:
            assert given in self._pred_distr
            population = list(self._pred_distr[given].keys())
            weights = list(self._pred_distr[given].values())
            return random.choices(population, weights)[0]

    def predict_sequence(self, length):
        """
        :param length: requested length of the sequence
        :return: list of tokens
        """
        sequence = list(self.predict())
        while len(sequence) < length:
            given = tuple(sequence[-self._n + 1:])
            if given in self._pred_distr:
                next_token = [self.predict(tuple(sequence[-self._n + 1:]))]
            else:
                print("Error...")
                next_token = self.predict()
            sequence.extend(next_token)
        return sequence

    def vectorize(self, codebook=None):
        """
        return a list of numbers reflecting the probability of each ngram
        :param codebook: ngrams
        :return: a list of numbers
        """
        if codebook is None:
            codebook = self._ngrams.keys()
        return [self._ngrams[ngram] / self.token_num for ngram in codebook]

    def union(self, other):
        """
        union another ngram model
        :param other: another ngram model
        :return: a combined model
        """
        new_model = deepcopy(self)
        for ngram in other._ngrams:
            new_model._ngrams[ngram] += other._ngrams[ngram]
        return new_model
    
    def intersect(self, other):
        """
        find the intersect of another ngram model
        :param other: another ngram model
        :return: None
        """
        new_model = deepcopy(self)
        for ngram in other._ngrams:
            new_model._ngrams[ngram] = other._ngrams[ngram] if ngram in new_model._ngrams else 0
        return new_model
    
    def subtract(self, other):
        """
        subtract another ngram model
        :param other: another ngram model
        :return: None
        """
        new_model = deepcopy(self)
        for ngram in other._ngrams:
            if ngram in new_model._ngrams:
                new_model._ngrams.pop(ngram)
        return new_model

    def perplexity(self, token_list):
        perplexity = 1
        h = []
        for token in token_list:
            h.append(token)
            if len(h) > self._n:
                h.pop(0)
            if len(h) == self._n:
                cur_p = (self._ngrams[tuple(h)] + 1) / (self._n_1grams[tuple(h[:-1])] + self.vocab_size)
                print(cur_p)
                perplexity *= cur_p
        return (1 / perplexity) ** (1/len(token_list))


        
if __name__ == "__main__":
    train_text = "<s> I would much rather eat pizza than ice cream . </s>".split()
    model = NGramModel(train_text, 2)
    print(model._ngrams)
    print(model.predict())
    print(model._pred_distr)
    print(model.predict_sequence(10))

    test_text = "<s> I love anchovies on my pizza . </s>".split()
    print(model.perplexity(test_text))
    print(model.vocab_size)
