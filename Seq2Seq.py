import tensorflow as tf
import numpy as np
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os

# Suppress Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def buildTrainingMatrices(convFile, vocabList, seqMaxLen):
    convoDict = np.load(convFile, allow_pickle=True).item()
    totalSamples = len(convoDict)
    encoderTrainData = np.zeros((totalSamples, seqMaxLen), dtype='int32')
    decoderTrainData = np.zeros((totalSamples, seqMaxLen), dtype='int32')

    for idx, (inputMsg, replyMsg) in enumerate(convoDict.items()):
        encVec = np.full((seqMaxLen), vocabList.index('<pad>'), dtype='int32')
        decVec = np.full((seqMaxLen), vocabList.index('<pad>'), dtype='int32')

        inputTokens = inputMsg.split()
        replyTokens = replyMsg.split()
        inLen = len(inputTokens)
        outLen = len(replyTokens)

        if (inLen > (seqMaxLen - 1) or outLen > (seqMaxLen - 1) or inLen == 0 or outLen == 0):
            continue

        # Encoder mapping
        for wi, word in enumerate(inputTokens):
            try:
                encVec[wi] = vocabList.index(word)
            except ValueError:
                encVec[wi] = 0
        encVec[wi + 1] = vocabList.index('<EOS>')

        # Decoder mapping
        for wj, word in enumerate(replyTokens):
            try:
                decVec[wj] = vocabList.index(word)
            except ValueError:
                decVec[wj] = 0
        decVec[wj + 1] = vocabList.index('<EOS>')

        encoderTrainData[idx] = encVec
        decoderTrainData[idx] = decVec

    # Drop all-zero rows
    decoderTrainData = decoderTrainData[~np.all(decoderTrainData == 0, axis=1)]
    encoderTrainData = encoderTrainData[~np.all(encoderTrainData == 0, axis=1)]

    return encoderTrainData.shape[0], encoderTrainData, decoderTrainData


def fetchTrainingBatch(xData, yData, bSize, seqMaxLen):
    randIdx = randint(0, numSamples - bSize - 1)
    encBatch = xData[randIdx:randIdx + bSize]
    decBatch = yData[randIdx:randIdx + bSize]

    reversedEnc = [list(reversed(example)) for example in encBatch]

    laggedTargets = []
    eosIdx = vocabulary.index('<EOS>')
    padIdx = vocabulary.index('<pad>')

    for seq in decBatch:
        eosPos = np.argwhere(seq == eosIdx)[0]
        shiftedSeq = np.roll(seq, 1)
        shiftedSeq[0] = eosIdx
        if eosPos != (seqMaxLen - 1):
            shiftedSeq[eosPos + 1] = padIdx
        laggedTargets.append(shiftedSeq)

    return np.asarray(reversedEnc).T.tolist(), decBatch.T.tolist(), np.asarray(laggedTargets).T.tolist()


def sentenceFromIds(predIds, vocabList):
    eosIdx = vocabList.index('<EOS>')
    padIdx = vocabList.index('<pad>')
    outputs, temp = [], ""

    for token in predIds:
        if (token[0] == eosIdx or token[0] == padIdx):
            if temp:
                outputs.append(temp.strip())
                temp = ""
        else:
            temp += vocabList[token[0]] + " "
    if temp:
        outputs.append(temp.strip())
    return outputs


def convertToTestInput(userMsg, vocabList, seqMaxLen):
    encVec = np.full((seqMaxLen), vocabList.index('<pad>'), dtype='int32')
    tokens = userMsg.lower().split()

    for idx, word in enumerate(tokens):
        try:
            encVec[idx] = vocabList.index(word)
        except ValueError:
            continue
    encVec[idx + 1] = vocabList.index('<EOS>')
    encVec = encVec[::-1]

    return [[num] for num in encVec]


# ------------------ Hyperparameters ------------------
batchSize = 24
maxSeqLen = 15
hiddenUnits = 112
embeddingDim = hiddenUnits
trainingSteps = 500000

# ------------------ Load vocab ------------------
with open("wordList.txt", "rb") as fp:
    vocabulary = pickle.load(fp)

vocabSize = len(vocabulary)

# Load embeddings
if os.path.isfile('embeddingMatrix.npy'):
    embeddings = np.load('embeddingMatrix.npy')
    embedDim = embeddings.shape[1]
else:
    embedDim = int(input("Embedding matrix missing. Enter embedding dimension: "))

padVec = np.zeros((1, embedDim), dtype='int32')
eosVec = np.ones((1, embedDim), dtype='int32')

if os.path.isfile('embeddingMatrix.npy'):
    embeddings = np.concatenate((embeddings, padVec, eosVec), axis=0)

vocabulary.extend(['<pad>', '<EOS>'])
vocabSize += 2

# ------------------ Load or create training data ------------------
if os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy'):
    Xtrain = np.load('Seq2SeqXTrain.npy')
    Ytrain = np.load('Seq2SeqYTrain.npy')
    numSamples = Xtrain.shape[0]
    print("Training matrices loaded")
else:
    numSamples, Xtrain, Ytrain = buildTrainingMatrices('conversationDictionary.npy', vocabulary, maxSeqLen)
    np.save('Seq2SeqXTrain.npy', Xtrain)
    np.save('Seq2SeqYTrain.npy', Ytrain)
    print("Training matrices built")


# ------------------ Seq2Seq Graph ------------------
tf.reset_default_graph()

encInputs = [tf.placeholder(tf.int32, shape=(None,)) for _ in range(maxSeqLen)]
decLabels = [tf.placeholder(tf.int32, shape=(None,)) for _ in range(maxSeqLen)]
decInputs = [tf.placeholder(tf.int32, shape=(None,)) for _ in range(maxSeqLen)]
usePrev = tf.placeholder(tf.bool)

encCell = tf.nn.rnn_cell.BasicLSTMCell(hiddenUnits, state_is_tuple=True)

decOutputs, decFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encInputs, decInputs, encCell, vocabSize, vocabSize, embeddingDim, feed_previous=usePrev)

predictions = tf.argmax(decOutputs, 2)

lossWeights = [tf.ones_like(label, dtype=tf.float32) for label in decLabels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decOutputs, decLabels, lossWeights, vocabSize)
trainOp = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss', loss)
mergedSumm = tf.summary.merge_all()
logDir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logDir, sess.graph)

# ------------------ Testing Strings ------------------
sampleInputs = ["whats up", "hi", "hey how are you", "what are you up to", "that dodgers game was awesome"]
zeroVec = np.zeros((1), dtype='int32')

# ------------------ Training Loop ------------------
for step in range(trainingSteps):
    encBatch, decBatch, decLagged = fetchTrainingBatch(Xtrain, Ytrain, batchSize, maxSeqLen)
    feedDict = {encInputs[t]: encBatch[t] for t in range(maxSeqLen)}
    feedDict.update({decLabels[t]: decBatch[t] for t in range(maxSeqLen)})
    feedDict.update({decInputs[t]: decLagged[t] for t in range(maxSeqLen)})
    feedDict.update({usePrev: False})

    curLoss, _, _ = sess.run([loss, trainOp, predictions], feed_dict=feedDict)

    if step % 50 == 0:
        print(f"Step {step} - Loss: {curLoss}")
        summary = sess.run(mergedSumm, feed_dict=feedDict)
        writer.add_summary(summary, step)

    if step % 25 == 0 and step != 0:
        testMsg = sampleInputs[randint(0, len(sampleInputs) - 1)]
        print("Input:", testMsg)
        testInputVec = convertToTestInput(testMsg, vocabulary, maxSeqLen)
        testDict = {encInputs[t]: testInputVec[t] for t in range(maxSeqLen)}
        testDict.update({decLabels[t]: zeroVec for t in range(maxSeqLen)})
        testDict.update({decInputs[t]: zeroVec for t in range(maxSeqLen)})
        testDict.update({usePrev: True})

        resultIds = sess.run(predictions, feed_dict=testDict)
        print("Bot:", sentenceFromIds(resultIds, vocabulary))

    if step % 10000 == 0 and step != 0:
        saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=step)
