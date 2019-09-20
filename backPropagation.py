import numpy as np
import math
import random

class backPropagation():
    input = 2
    hidden = 2
    output = 1
    error = 0

    def __init__(self, trainSet, error, rate):
        self.trainSet = trainSet
        self.error = error
        self.rate = rate
        self.inWeight = np.zeros((self.input, self.hidden))
        for i in range(self.input):
            for j in range(self.hidden):
                self.inWeight[i][j] = random.uniform(-1, 1)
        self.hideWeight = np.zeros((self.hidden, self.output))
        for i in range(self.hidden):
            for j in range(self.output):
                self.hideWeight[i][j] = random.uniform(-1, 1)

    def train(self):
        hideValue = np.zeros(self.hidden)
        outValue = np.zeros(self.output)
        inOut = np.zeros(self.hidden)
        hideOut = np.zeros(self.output)
        hideErr = np.zeros(self.hidden)
        e = 0.0
        delta_inWeight = np.zeros((self.input, self.hidden))
        delta_hideWeight = np.zeros((self.hidden, self.output))

        for index in range(len(self.trainSet)):
            # calculate hide value
            for i in range(self.hidden):
                tmp = 0.0
                for j in range(self.input):
                    tmp += self.trainSet[index][j] * self.inWeight[j][i]
                inOut[i] = tmp
                hideValue[i] = self.activateFunc(inOut[i])
            # calculate output value
            for i in range(self.output):
                tmp = 0.0
                for j in range(self.hidden):
                    tmp += hideValue[j] * self.hideWeight[j][i]
                hideOut[i] = tmp
                outValue[i] = self.activateFunc(hideOut[i])
            # calculate error for each dataset
            err = (self.trainSet[index][2] - outValue[0]) * (1 - outValue[0]) * outValue[0]
            er = self.trainSet[index][2] - outValue[0]
            #print(outValue[0])
            e += 0.5 * pow(er, 2)
            # calculate delta hide weight
            for i in range(self.hidden):
                tmpl = []
                tmp = self.rate * err * hideValue[i]
                tmpl.append(tmp + delta_hideWeight[i][0])
                delta_hideWeight[i] = tmpl
            # calculate hide error
            for i in range(self.hidden):
                hideErr[i] = err * self.hideWeight[i][0] * (1 - hideValue[i]) * hideValue[i]
            # calculate delta input weight
            for i in range(self.input):
                tmpl = []
                for j in range(self.hidden):
                    tmp = self.rate * hideErr[j] * self.trainSet[index][i]
                    tmpl.append(tmp + delta_inWeight[i][j])
                delta_inWeight[i] = tmpl

        # add delta input weight to input weight
        for i in range(self.input):
            tmpl = []
            for j in range(self.hidden):
                tmp = self.inWeight[i][j] + delta_inWeight[i][j]
                tmpl.append(tmp)
            self.inWeight[i] = tmpl
        # add delta hide weight to hide weight
        for i in range(self.hidden):
            tmpl = []
            for j in range(self.output):
                tmp = self.hideWeight[i][j] + delta_hideWeight[i][j]
                tmpl.append(tmp)
            self.hideWeight[i] = tmpl

        self.error = e / 4

    def activateFunc(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def get_inWeight(self):
        return self.inWeight

    def get_hideWeight(self):
        return self.hideWeight
