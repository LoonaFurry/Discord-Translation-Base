import discord
import torch
import re
import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, NllbTokenizer
from discord.ext import commands

# Set up the Discord client
client = discord.Client(intents=discord.Intents.default())

# Check for GPU availability and set up the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set up the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(device)

# Set up the Roberta model and tokenizer for Swear And Hate Speech Detection
hate_speech_tokenizer = AutoTokenizer.from_pretrained('Hate-speech-CNERG/dehatebert-mono-english')
hate_speech_model = AutoModelForSequenceClassification.from_pretrained('Hate-speech-CNERG/dehatebert-mono-english')

# Set up the model generation parameters
max_length = 2048
temperature = 1000
repetition_penalty = 1.0
num_beams = 5
no_repeat_ngram_size = 3
early_stopping = True

# Set up chat log file
chat_log_file = 'chat_log.txt'

# Load conversation history from file
conversation_history = []
try:
    with open(chat_log_file, 'r', encoding='utf-8') as f:
        for line in f:
            conversation_history.append(line.strip())
except FileNotFoundError:
    pass


# Function to detect hate speech
def detect_hate_speech(input_text):
    # Tokenize the input text
    input_ids = hate_speech_tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Set the attention mask to 1 for all input tokens
    attention_mask = torch.ones_like(input_ids)

    # Classify the input text using Roberta
    outputs = hate_speech_model(input_ids=input_ids, attention_mask=attention_mask)

    # Get the predicted label
    predicted_label = outputs.logits.argmax().item()

    return predicted_label


# Function to generate a response from the chatbot
async def generate_response(user, input_text):
    # Remove the user's mention from the input text
    input_text = re.sub(r"<@\!?\d+>", "", input_text).strip()

    # Check if the input text contains hate speech
    hate_speech_label = detect_hate_speech(input_text)
    if hate_speech_label == 1:
        response = "I'm sorry, I cannot respond to this message as it contains hate speech or swear words."
        return f"{user.mention}: {response}"

    # Prepend the translation phrase to the input text
   # input_text = "translate to English and make grammar check" + input_text

    # Append the translation phrase to the input text
    input_text += " translate to English"

    # Add the user's mention to the context
    context = f"{user.name}: "

    # Tokenize the input text and context
    input_ids = tokenizer.encode(f"{context}{input_text}\n", return_tensors="pt").to(device)

    # Set the attention mask to 1 for all input tokens
    attention_mask = torch.ones_like(input_ids)

    # Generate a response using T5
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
        use_cache=True
    )

    # Decode the response and remove the bot mention
    response = tokenizer.decode(output[0], skip_special_tokens=True).replace(client.user.mention, '').split('\n')[0]

    # Remove the user's name from the response
    response = response.replace(user.name, '')

    # Check if the response contains only colons or whitespace
    if all(c in (':', ' ') for c in response):
        response = "I'm sorry, I don't understand that."

    # If the response is empty or contains only whitespace, return a generic error message
    if not response or response.strip() == '':
        response = "I'm sorry, I don't have a response for that."

    # Save the conversation history to the chat log file and add it to the conversation history list
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(chat_log_file, 'a', encoding='utf-8') as f:
        f.write(f"{now} - {user.name}: {input_text}\n")
        f.write(f"{now} - Bot: {response}\n")
    conversation_history.append(f"{now} - {user.name}: {input_text}")
    conversation_history.append(f"{now} - Bot: {response}")

    return f"{user.mention}: {response}"


# Function to handle incoming messages
async def handle_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the bot is mentioned in the message
    if client.user.mentioned_in(message):
        # Generate a response
        response = await generate_response(message.author, message.content)

        # Send the response to the channel
        await message.channel.send(response)


# Event handler for bot startup
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    # Set the bot's activity and status
    activity = discord.Activity(
        type=discord.ActivityType.playing,
        name="Language Translator Bot | Learn New Language"
    )
    await client.change_presence(activity=activity, status=discord.Status.online)


# Function to handle incoming messages
@client.event
async def on_message(message):
    # Handle the message
    await handle_message(message)

# Run the bot with your Discord bot token
client.run("your-token-here")