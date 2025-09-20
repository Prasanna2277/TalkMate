# TalkMate

**TalkMate** is a personal chatbot trained to talk like you. Using your past conversation logs, it leverages a Sequence-to-Sequence (Seq2Seq) model to generate responses that mimic your style.

## Overview

This project allows you to train a chatbot that can respond in a conversational style similar to yours. It uses deep learning techniques (Seq2Seq with embeddings) on your historical chat data from multiple platforms. The chatbot can later be deployed through a web interface or integrated into messaging platforms.

## Requirements and Installation

You’ll need the following Python libraries:

* **TensorFlow** version 1.0 or later
* **NumPy**
* **Pandas**
* **Scikit-learn**

Install them via pip if you don’t already have them:

```bash
pip install tensorflow numpy pandas scikit-learn
```

## Preparing Your Data

TalkMate uses your past conversation logs to learn your conversational style. Supported sources include:

* **Facebook**: Download your messages from [Facebook Data Download](https://www.facebook.com/help/131112897028467). Convert `messages.htm` to text using [fbchat-archive-parser](https://github.com/ownaginatious/fbchat-archive-parser):

```bash
pip install fbchat-archive-parser
fbcap ./messages.htm > fbMessages.txt
```

* **LinkedIn**: Download your data from [LinkedIn Data Export](https://www.linkedin.com/psettings/member-data). Copy the `inbox.csv` file into your project folder.

* **Google Hangouts**: Export your chat history via [Google Takeout](https://takeout.google.com/) and convert JSON files to text.

* **Discord**: Export logs in `.txt` format using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).

* **WhatsApp**: Export chats via the “Email Chat” option in the app. Convert `.txt` logs to `.csv` if needed, or place `.txt` files in `WhatsAppChatLogs` folder.

Folder structure should look like:

```
TalkMate/
├── FacebookMessages/
├── LinkedInMessages/
├── HangoutsMessages/
├── DiscordChatLogs/
├── WhatsAppChatLogs/
```

## Creating the Dataset

Run the following script to convert your conversation logs into training data:

```bash
python createDataset.py
```

This will generate:

* `conversationDictionary.npy` – Contains (friend message, your response) pairs.
* `conversationData.txt` – Unified text of all conversations.

## Word Embeddings

Next, generate word vectors (Word2Vec) or let the Seq2Seq model create embeddings on the fly:

```bash
python Word2Vec.py
```

Outputs include:

* `Word2VecXTrain.npy` & `Word2VecYTrain.npy`
* `wordList.txt`
* `embeddingMatrix.npy`

## Training the Seq2Seq Model

```bash
python Seq2Seq.py
```

Generates:

* `Seq2SeqXTrain.npy` & `Seq2SeqYTrain.npy`
* `.ckpt` files – Saved TensorFlow models.

## Deploying the Chatbot

1. Create a Flask app (e.g., `app.py`) that loads your saved Seq2Seq model.
2. Expose a `/prediction` endpoint that takes a user message as input and returns a generated response.
3. Optionally, integrate with a Node.js/Express app for Facebook Messenger or other chat platforms.

### Hosting Options

* **Heroku**: Deploy your Flask app or Node.js integration.
* **Other cloud providers**: AWS, GCP, or Azure.

## Sample Usage

```python
from Seq2SeqHelper import convertToTestInput, sentenceFromIds, vocabulary, sess, encInputs, decLabels, decInputs, usePrev, predictions

user_msg = "Hey, how are you?"
test_input = convertToTestInput(user_msg, vocabulary, maxSeqLen)
test_dict = {encInputs[t]: test_input[t] for t in range(maxSeqLen)}
test_dict.update({decLabels[t]: zeroVec for t in range(maxSeqLen)})
test_dict.update({decInputs[t]: zeroVec for t in range(maxSeqLen)})
test_dict.update({usePrev: True})

response_ids = sess.run(predictions, feed_dict=test_dict)
print("TalkMate:", sentenceFromIds(response_ids, vocabulary))
```

## Notes

* Ensure your dataset is cleaned and in the correct format.
* Training may take several hours depending on your dataset size and GPU availability.
* Save checkpoints frequently during training.

