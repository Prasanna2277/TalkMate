'use strict'

const express = require('express')
const bodyParser = require('body-parser')
const request = require('request')
const app = express()

// Server port
app.set('port', process.env.PORT || 5000)

// Middleware
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())

// Root route
app.get('/', (req, res) => {
	res.send('Hello world, I am a chatbot')
})

// Facebook webhook verification
app.get('/webhook/', (req, res) => {
	if (req.query['hub.verify_token'] === 'verify_token_secret') {
		res.send(req.query['hub.challenge'])
	} else {
		res.send('Error, invalid verification token')
	}
})

const BOT_ID = 1960455644201170
const PAGE_ACCESS_TOKEN = process.env.FB_PAGE_ACCESS_TOKEN

// Webhook to handle incoming messages
app.post('/webhook/', (req, res) => {
	let incomingEvents = req.body.entry[0].messaging

	for (let i = 0; i < incomingEvents.length; i++) {
		let messageEvent = incomingEvents[i]
		let userId = messageEvent.sender.id

		// Only reply if message exists and not sent by bot itself
		if (messageEvent.message && messageEvent.message.text && userId !== BOT_ID) {
			let userMessage = messageEvent.message.text

			// Send message to Flask server for prediction
			request({
				url: 'https://flask-server-seq2seq-chatbot.herokuapp.com/prediction',
				method: 'POST',
				body: { message: userMessage.substring(0, 200) },
				headers: { 'User-Agent': 'request' },
				json: true
			}, (err, response, body) => {
				if (!err && response && response.body) {
					sendMessageToUser(userId, response.body)
				} else {
					console.log('Error in response from Flask server:', err || response)
				}
			})
		}
	}

	res.sendStatus(200)
})

// Function to send text message back to user
function sendMessageToUser(userId, replyText) {
	let payload = {
		recipient: { id: userId },
		message: { text: replyText }
	}

	request({
		url: 'https://graph.facebook.com/v2.6/me/messages',
		qs: { access_token: PAGE_ACCESS_TOKEN },
		method: 'POST',
		json: payload
	}, (err, response, body) => {
		if (err) {
			console.log('Error sending messages: ', err)
		} else if (response.body && response.body.error) {
			console.log('Facebook API Error: ', response.body.error)
		}
	})
}

// Start server
app.listen(app.get('port'), () => {
	console.log('Bot server running on port', app.get('port'))
})
