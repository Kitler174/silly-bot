import discord
from discord import app_commands
import os
import pandas as pd
from pypresence import AioPresence
import asyncio
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

client_id = "1213032690425667614"  # Twój client_id z Discord Developer Portal
folder_path = "learn-files"
token = "MTIxMzAzMjY5MDQyNTY2NzYxNA.GVJqik.Qz3o-NihESSzm0837kbUWj2v0bK5Pp1pPN-JD0"
learn_mode_active = False
talk_mode_active = False
MAX_RECORDS = 1000
number = 0
with open("plik.txt", "r") as file:
    number = file.read().strip()
current_file = f"{folder_path}/learn_data_{number}.csv"  # Ścieżka do początkowego pliku CSV

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
RPC = AioPresence(client_id)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

def create_csv_file(file_path):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["id", "message", "response"])  # Kolumny w pliku CSV
        df.to_csv(file_path, index=False)

# Funkcja dodająca wiadomość do pliku CSV
def add_message_to_csv(message, response):
    global current_file
    df = pd.read_csv(current_file)

    # Jeśli liczba rekordów przekroczy MAX_RECORDS, tworzymy nowy plik z numerem sekwencyjnym
    if len(df) >= MAX_RECORDS:
        # Znalezienie najwyższego numeru pliku w folderze
        existing_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        max_number = 0
        for file in existing_files:
            try:
                file_number = int(file.split('_')[-1].split('.')[0])  # Pobieramy numer pliku
                max_number = max(max_number, file_number)
            except ValueError:
                continue
        
        new_file_number = max_number + 1  # Nowy numer pliku
        with open("plik.txt", "w") as file:
            file.write(str(new_file_number))
        current_file = f"{folder_path}/learn_data_{new_file_number}.csv"
        create_csv_file(current_file)
        df = pd.read_csv(current_file)  # Odczytujemy nowy plik

    # Ustalanie nowego ID na podstawie liczby rekordów
    new_id = len(df) + 1
    new_record = pd.DataFrame([[new_id, message, response]], columns=["id", "message", "response"])

    # Dodanie nowego rekordu do DataFrame
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(current_file, index=False)

# Funkcja licząca liczbę rekordów w pliku CSV
def get_record_count():
    df = pd.read_csv(current_file)
    return len(df)

# Funkcja licząca liczbę plików i rekordów w folderze
def count_files_and_records(folder_path):
    file_count = 0
    total_records = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # Sprawdzamy pliki CSV
            file_count += 1
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                total_records += len(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return file_count, total_records

# Komendy Discorda
@tree.command(name="ping", description="Replies with the bot's latency")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")
    
@tree.command(name="math", description="Solve math problems")
async def math(interaction: discord.Interaction, txt: str = ""):
    try:
        txt = eval(txt)
    except Exception:
        txt = ""
    await interaction.response.send_message(txt)

@tree.command(name="talkingon", description="Enables talking mode")
async def talkingon(interaction: discord.Interaction):
    global talk_mode_active
    txt = "Talking mode has been enabled!"
    if talk_mode_active:
        txt = "Talking mode has been disabled!"
        talk_mode_active = False
    else:
        talk_mode_active = True
    await interaction.response.send_message(txt)

@tree.command(name="learnmode", description="Activates learning mode")
async def learnmode(interaction: discord.Interaction):
    global learn_mode_active
    if learn_mode_active:
        learn_mode_active = False
    else:
        learn_mode_active = True
    learn_mode_active = True
    files, records = count_files_and_records(folder_path)
    embed = discord.Embed(title="Info", description="Current information about collected data", color=0xff0000)
    embed.set_author(
        name="Mateusz Błaszczyk - GitHub",
        url="https://github.com/Kitler174",
        icon_url="https://cdn.pixabay.com/photo/2022/01/30/13/33/github-6980894_1280.png"
    )
    embed.add_field(name="Learning mode status:", value="Enabled", inline=True)
    embed.add_field(name="Activated by:", value=interaction.user.mention, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=False)  # Invisible field
    embed.add_field(name="Number of files:", value=files, inline=True)
    embed.add_field(name="Amount of data:", value=records, inline=True)
    embed.set_footer(text="Server ID: " + str(interaction.guild.id))

    await interaction.response.send_message(embed=embed)

# Aktualizacja Rich Presence
async def update_rich_presence():
    await RPC.connect()
    while True:
        await RPC.update(
            state="and drinking vodka",
            details="Drinking Acid",
            start=int(time.time()),
            large_image="embedded_cover",
            small_image="small_image_key",
            large_text="Big image description",
            small_text="Small image description",
        )
        await asyncio.sleep(15)

@client.event
async def on_ready():
    create_csv_file(current_file)  # Upewnij się, że plik CSV jest stworzony, jeśli nie istnieje
    asyncio.create_task(update_rich_presence())
    await client.change_presence(
        activity=discord.Game(name="Drinking Acid")
    )
    await tree.sync(guild=discord.Object(id=1213035288495136768))
    print(f'Logged in as {client.user} (ID: {client.user.id})')


def generate_response(message):
    # Tokenizowanie wiadomości
    inputs = tokenizer.encode(message, return_tensors="pt")

    # Generowanie odpowiedzi
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

    # Dekodowanie wygenerowanej odpowiedzi
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
@client.event
async def on_message(message):
    global learn_mode_active
    global talk_mode_active
    if message.author.bot:
        return  # Ignorowanie wiadomości od bota

    if talk_mode_active:
        # Dodanie wiadomości do pliku CSV
        response = generate_response(message.content)
        if learn_mode_active:
            add_message_to_csv(message.content,response)

            # Sprawdzenie, czy trzeba przełączyć na nowy plik (jeśli liczba rekordów przekroczy MAX_RECORDS)
            if get_record_count() >= MAX_RECORDS:
                global current_file
                current_file = f"{folder_path}/learn_data_{int(time.time())}.csv"  # Zmieniamy nazwę pliku
                create_csv_file(current_file)
        embed = discord.Embed(title="Response", description="", color=0xfff000)
        embed.set_author(
            name="Mateusz Błaszczyk - GitHub",
            url="https://github.com/Kitler174",
            icon_url="https://cdn.pixabay.com/photo/2022/01/30/13/33/github-6980894_1280.png"
        )
        embed.add_field(name="Learning mode status:", value=learn_mode_active, inline=True)
        embed.add_field(name="\u200b", value="\u200b", inline=False)  # Invisible field
        embed.add_field(name="Message:", value=message.content, inline=False)
        embed.add_field(name="Response:", value=response, inline=False)
        embed.set_footer(text="Server ID: " + str(message.guild.id))
        await message.channel.send(embed=embed)

client.run(token)
