import asyncio
import io
import os
import re
import sys
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageSequence
import edge_tts
import aiohttp

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

EPISODE_EMOJI = "<:emoji_name:emoji_id>"
OVERLAY_FILENAME = "overlay.png"

DISABLED_MODELS = {}

DEFAULT_CHARACTERS = {
    "Fred": {
        "name": "Fred Figglehorn",
        "avatar": "https://i.pinimg.com/1200x/72/60/08/726008f18672dfc798180d1185d977ae.jpg",
        "color": discord.Color.gold(),
        "voice": "en-GB-ThomasNeural",
    },
    "Kevin": {
        "name": "Kevin",
        "avatar": "https://wertigo.ru/api/shared_files/b7bef8c8-99fe-415f-a1d2-de1f758445eb/files/b7bef8c8-99fe-415f-a1d2-de1f758445eb/preview",
        "color": discord.Color.blue(),
        "voice": "en-US-AndrewNeural",
    },
    "Angry Fred": {
        "name": "Angry Fred",
        "avatar": "https://iili.io/CtW9xWX.jpg",
        "color": discord.Color.red(),
        "voice": "en-US-ChristopherNeural",
    },
}

CHARACTERS = DEFAULT_CHARACTERS.copy()
GUILD_USER_CHARACTERS = {}

# put fish audio voices
FISH_AUDIO_VOICES = {
    "Fred": "test",
    "Kevin": "test2",
    "Angry Fred": "test3",
}


class ParodyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.episode_queue = asyncio.Queue()
        self.queue_list = []  # List tracking metadata for active/queued items
        self.worker_task = None

    async def setup_hook(self):
        await self.tree.sync()
        print("Slash commands synced successfully.")
        self.worker_task = asyncio.create_task(self.episode_queue_worker())

    async def episode_queue_worker(self):
        """Processes episode creation requests sequentially in the background."""
        while True:
            task_func, interaction, metadata = await self.episode_queue.get()
            try:
                metadata["status"] = "Generating"
                await task_func(interaction)
            except Exception as e:
                print(f"Error processing episode task: {e}")
                try:
                    await interaction.followup.send(
                        f"An error occurred during episode processing: {e}"
                    )
                except Exception:
                    pass
            finally:
                if metadata in self.queue_list:
                    self.queue_list.remove(metadata)
                self.episode_queue.task_done()


bot = ParodyBot()


@bot.event
async def on_ready():
    activity = discord.Game(name="made with Python!")
    await bot.change_presence(status=discord.Status.online, activity=activity)
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


async def generate_tts_audio(text: str, voice: str) -> io.BytesIO:
    communicate = edge_tts.Communicate(text, voice)
    audio_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.write(chunk["data"])
    audio_data.seek(0)
    return audio_data


async def generate_fish_audio_tts(text: str, reference_id: str) -> io.BytesIO:
    url = "https://openrouter.ai/api/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "fish-audio/s2.1-pro-free:free",
        "input": text,
        "voice": reference_id,
        "response_format": "mp3",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                audio_data = io.BytesIO(await resp.read())
                audio_data.seek(0)
                return audio_data
            else:
                err_text = await resp.text()
                raise Exception(f"<:emoji_name:emoji_id> OpenRouter TTS Error ({resp.status}): {err_text}")


async def request_openrouter_script(prompt: str, model_id: str) -> str:
    if model_id.startswith("gemini-"):
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings,
                temperature=0.7,
            ),
        )
        return response.text.strip() if response.text else ""

    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    err_body = await resp.text()
                    raise Exception(f"<:emoji_name:emoji_id> OpenRouter LLM Error ({resp.status}): {err_body}")


@bot.tree.command(name="restart", description="Restart the bot (Bot owner only).")
async def restart(interaction: discord.Interaction):
    app_info = await bot.application_info()
    if interaction.user.id != app_info.owner.id:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    await interaction.response.send_message("Restarting bot...", ephemeral=True)
    await bot.close()
    os.execv(sys.executable, [sys.executable] + sys.argv)


@bot.tree.command(name="add_user_character", description="Add server users in the channel as characters for this server's episodes.")
@app_commands.describe(user="Select a specific user (leave blank to pick active users from channel history)")
async def add_user_character(interaction: discord.Interaction, user: discord.Member = None):
    if not interaction.guild:
        await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
        return

    guild_id = interaction.guild.id
    if guild_id not in GUILD_USER_CHARACTERS:
        GUILD_USER_CHARACTERS[guild_id] = {}

    if user:
        added_members = [user]
    else:
        await interaction.response.defer(ephemeral=True)
        added_members = []
        async for msg in interaction.channel.history(limit=50):
            if isinstance(msg.author, discord.Member) and not msg.author.bot:
                if msg.author not in added_members:
                    added_members.append(msg.author)

    if not added_members:
        msg_text = "No non-bot users found in recent channel activity."
        if interaction.response.is_done():
            await interaction.followup.send(msg_text, ephemeral=True)
        else:
            await interaction.response.send_message(msg_text, ephemeral=True)
        return

    added_names = []
    for member in added_members:
        display_name = member.display_name
        GUILD_USER_CHARACTERS[guild_id][display_name] = {
            "name": display_name,
            "avatar": member.display_avatar.url,
            "color": member.color if member.color != discord.Color.default() else discord.Color.blue(),
            "voice": "en-GB-ThomasNeural",
        }
        added_names.append(display_name)

    response_text = f"Successfully added **{len(added_names)}** local user character(s) to this server: {', '.join(added_names)}"
    if interaction.response.is_done():
        await interaction.followup.send(response_text)
    else:
        await interaction.response.send_message(response_text)


@bot.tree.command(name="shutdown", description="Shutdown the bot (Bot owner only).")
async def shutdown(interaction: discord.Interaction):
    app_info = await bot.application_info()
    if interaction.user.id != app_info.owner.id:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    await interaction.response.send_message("Shutting down bot...", ephemeral=True)
    await bot.close()
    sys.exit(0)


MODEL_CHOICES = [
    app_commands.Choice(name="Gemini 3.1 Flash Lite (Fast, ratelimited)", value="gemini-3.1-flash-lite"),
    app_commands.Choice(name="Gemini 3.5 Flash Lite (Lightweight)", value="gemini-3.5-flash-lite"),
    app_commands.Choice(name="Gemini 3.7 Flash", value="gemini-3.7-flash"),
    app_commands.Choice(name="Gemini 3.8 Flash (NEW)", value="gemini-3.8-flash"),
    app_commands.Choice(name="Gemini 3.5 Flash", value="gemini-3.5-flash"),
    app_commands.Choice(name="MiniMax M2.7", value="minimax/minimax-m2.7"),
    app_commands.Choice(name="MiniMax M3", value="minimax/minimax-m3"),
    app_commands.Choice(name="NVIDIA: Nemotron 3 Super (free)", value="nvidia/nemotron-3-super-120b-a12b:free"),
    app_commands.Choice(name="NVIDIA: Nemotron 3 Nano Omni (free)", value="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    app_commands.Choice(name="NVIDIA: Nemotron 3 Ultra (free)", value="nvidia/nemotron-3-ultra-550b-a55b:free"),
    app_commands.Choice(name="NVIDIA: Nemotron 3.5 Lightning (free)", value="nvidia/nemotron-3.5-lightning:free"),
    app_commands.Choice(name="inclusionAI: Ling 3.0 Flash Sante (free)", value="inclusionai/ling-3.0-flash-sante:free"),
    app_commands.Choice(name="inclusionAI: Ling 3.0 Flash Fin (free)", value="inclusionai/ling-3.0-flash-fin:free"),
]


@bot.tree.command(name="remove_user_character", description="Remove custom user characters from this server's episode pool.")
@app_commands.describe(user="Select a specific user character to remove (leave blank to clear all user characters in this server)")
async def remove_user_character(interaction: discord.Interaction, user: discord.Member = None):
    if not interaction.guild:
        await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
        return

    guild_id = interaction.guild.id
    server_chars = GUILD_USER_CHARACTERS.get(guild_id, {})

    if not server_chars:
        await interaction.response.send_message("No custom user characters registered in this server.", ephemeral=True)
        return

    if user:
        target_name = user.display_name
        if target_name in server_chars:
            del server_chars[target_name]
            await interaction.response.send_message(f"Removed character **{target_name}** from this server.")
        else:
            await interaction.response.send_message(f"Character **{target_name}** is not in this server's user character list.", ephemeral=True)
    else:
        removed_names = list(server_chars.keys())
        GUILD_USER_CHARACTERS[guild_id].clear()
        await interaction.response.send_message(f"Cleared **{len(removed_names)}** local user character(s): {', '.join(removed_names)}")


@bot.tree.command(name="disable_model", description="Disable a model from being used in /episode (Owner only).")
@app_commands.describe(
    model="Select the model to disable",
    reason="The reason for disabling this model"
)
@app_commands.choices(model=MODEL_CHOICES)
async def disable_model(
    interaction: discord.Interaction,
    model: app_commands.Choice[str],
    reason: str = "Maintenance / Temporarily disabled."
):
    app_info = await bot.application_info()
    if interaction.user.id != app_info.owner.id:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    DISABLED_MODELS[model.value] = reason
    await interaction.response.send_message(
        f"Disabled model `{model.name}` (`{model.value}`).\n**Reason:** {reason}",
        ephemeral=True
    )


@bot.tree.command(name="enable_model", description="Re-enable a disabled model (Owner only).")
@app_commands.describe(model="Select the model to enable")
@app_commands.choices(model=MODEL_CHOICES)
async def enable_model(
    interaction: discord.Interaction,
    model: app_commands.Choice[str]
):
    app_info = await bot.application_info()
    if interaction.user.id != app_info.owner.id:
        await interaction.response.send_message(
            "Only the bot owner can use this command.", ephemeral=True
        )
        return

    if model.value in DISABLED_MODELS:
        del DISABLED_MODELS[model.value]
        await interaction.response.send_message(
            f"Re-enabled model `{model.name}` (`{model.value}`).",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            f"Model `{model.name}` is not currently disabled.",
            ephemeral=True
        )


@bot.tree.command(name="queue", description="Check the details of the active and queued episodes.")
async def show_queue(interaction: discord.Interaction):
    if not bot.queue_list:
        await interaction.response.send_message("The episode queue is currently empty!")
        return

    embed = discord.Embed(
        title="🎬 Episode Generation Queue",
        color=discord.Color.blurple()
    )

    for idx, item in enumerate(bot.queue_list):
        status_str = f"⚡ **{item['status']}**" if item['status'] == "Generating" else f"⏳ **Position #{idx}**"
        
        field_value = (
            f"**Requested by:** {item['user']}\n"
            f"**Model:** `{item['model']}`\n"
            f"**Turns:** {item['turns']} | **TTS:** {item['tts']}\n"
            f"**Status:** {status_str}"
        )
        embed.add_field(
            name=f"{idx + 1}. Topic: {item['topic']}",
            value=field_value,
            inline=False
        )

    embed.set_footer(text=f"Total items in process/queue: {len(bot.queue_list)}")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status_models", description="Check the status (Enabled/Disabled) of all supported AI models.")
async def status_models(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Model Status Overview",
        color=discord.Color.blue()
    )

    for choice in MODEL_CHOICES:
        model_id = choice.value
        model_name = choice.name

        if model_id in DISABLED_MODELS:
            reason = DISABLED_MODELS[model_id]
            status_text = f"🔴 **Disabled**\n**Reason:** {reason}"
        else:
            status_text = "🟢 **Enabled**"

        embed.add_field(
            name=model_name,
            value=f"`{model_id}`\n{status_text}",
            inline=False
        )

    await interaction.response.send_message(embed=embed)


async def run_episode_job(
    interaction: discord.Interaction,
    topic: str,
    turns: int,
    chosen_model: str,
    tts_engine: str,
    include_user_chars: bool,
    guild_id: int,
):
    await interaction.edit_original_response(content="Generating episode script... [10%]")

    active_characters = DEFAULT_CHARACTERS.copy()
    if include_user_chars and guild_id and guild_id in GUILD_USER_CHARACTERS:
        active_characters.update(GUILD_USER_CHARACTERS[guild_id])

    available_chars = ", ".join(active_characters.keys())
    format_example = "\n".join([f"{name}: [dialogue]" for name in list(active_characters.keys())[:3]])

    script_prompt = f"""
    Write a short parody episode script.
    Topic: {topic}
    
    Characters available: {available_chars}
    Strict limit: Exactly {turns} total dialogue turns. Ensure every available character speaks at least once.
    Format strictly as (DO NOT put brackets around character names):
    {format_example}
    """

    try:
        await interaction.edit_original_response(
            content=f"Requesting script via {chosen_model}... [30%]"
        )
        raw_script = await request_openrouter_script(script_prompt, chosen_model)

        if not raw_script:
            await interaction.followup.send("<:emoji_name:emoji_id> Content blocked or empty response.")
            return

    except Exception as e:
        await interaction.followup.send(f"<:emoji_name:emoji_id> Failed to generate script ({chosen_model}): {e}")
        return

    await interaction.edit_original_response(content="Formatting script embeds... [60%]")

    embeds = []
    files = []

    header_embed = discord.Embed(
        title=f"{EPISODE_EMOJI} EPISODE: {topic.upper()}",
        description=(
            f"*A {turns}-turn parody scene starring {available_chars}.*\n\n"
            f"🤖 **Model:** `{chosen_model}`\n"
            f"💬 **Turns:** {turns}\n"
            f"🔊 **TTS Engine:** `{tts_engine}`"
        ),
        color=discord.Color.dark_embed(),
    )
    embeds.append(header_embed)

    script_lines = [l.strip() for l in raw_script.split("\n") if l.strip()]
    total_lines = len(script_lines)
    line_count = 0

    for idx, line in enumerate(script_lines):
        match = re.match(r"^([^:]+)\s*:\s*(.*)$", line)
        if match:
            speaker_key = re.sub(r"[\[\]]", "", match.group(1)).strip()
            dialogue = match.group(2).strip()

            char_data = active_characters.get(
                speaker_key,
                {
                    "name": speaker_key,
                    "avatar": None,
                    "color": discord.Color.light_grey(),
                    "voice": "en-GB-ThomasNeural",
                },
            )

            line_embed = discord.Embed(
                description=dialogue, color=char_data.get("color", discord.Color.light_grey())
            )
            line_embed.set_author(
                name=char_data.get("name", speaker_key),
                icon_url=char_data.get("avatar"),
            )
            embeds.append(line_embed)

            if tts_engine != "none" and len(files) < 15:
                line_count += 1
                progress = 60 + int((idx / max(total_lines, 1)) * 30)
                await interaction.edit_original_response(
                    content=f"Generating {tts_engine.upper()} audio line {line_count}/{total_lines}... [{progress}%]"
                )
                try:
                    if tts_engine == "fish_audio":
                        ref_id = FISH_AUDIO_VOICES.get(
                            speaker_key, "test4"
                        )
                        audio_stream = await generate_fish_audio_tts(
                            dialogue, ref_id
                        )
                    else:
                        speaker_voice = char_data.get(
                            "voice", "en-GB-ThomasNeural"
                        )
                        audio_stream = await generate_tts_audio(
                            dialogue, speaker_voice
                        )

                    files.append(
                        discord.File(
                            audio_stream,
                            filename=f"line_{line_count}_{speaker_key}.mp3",
                        )
                    )
                except Exception as tts_err:
                    print(f"<:emoji_name:emoji_id> TTS generation failed for line {line_count}: {tts_err}")

    if len(embeds) > 10:
        embeds = embeds[:10]

    if len(embeds) <= 1:
        await interaction.followup.send("Script generation produced no dialogue.")
        return

    await interaction.edit_original_response(content="Uploading episode... [95%]")

    if files:
        await interaction.followup.send(embeds=embeds, files=files)
    else:
        await interaction.followup.send(embeds=embeds)

    await interaction.edit_original_response(content="Episode generation complete!")


@bot.tree.command(name="episode", description="Generate an AI parody episode.")
@app_commands.describe(
    topic="The topic for the episode",
    turns="Number of dialogue turns (3 to 10)",
    model="Choose AI Model provider and version",
    tts="Select Text-to-Speech Provider",
    include_user_chars="Include added local user characters in the script (Default: False)"
)
@app_commands.choices(
    model=MODEL_CHOICES,
    tts=[
        app_commands.Choice(name="Disabled", value="none"),
        app_commands.Choice(name="Edge-TTS (Free)", value="edge_tts"),
        app_commands.Choice(name="Fish Audio (Custom Voices)", value="fish_audio"),
    ],
)
async def episode(
    interaction: discord.Interaction,
    topic: str,
    turns: app_commands.Range[int, 3, 10] = 6,
    model: app_commands.Choice[str] = None,
    tts: app_commands.Choice[str] = None,
    include_user_chars: bool = False,
):
    chosen_model = model.value if model else "gemini-3.5-flash-lite"
    tts_engine = tts.value if tts else "none"
    guild_id = interaction.guild.id if interaction.guild else None

    if chosen_model in DISABLED_MODELS:
        reason = DISABLED_MODELS[chosen_model]
        await interaction.response.send_message(
            f"The selected model (`{chosen_model}`) is currently disabled.\n**Reason:** {reason}",
            ephemeral=True,
        )
        return

    if chosen_model.startswith("gemini-") and not gemini_client:
        await interaction.response.send_message(
            "Gemini API key is missing.", ephemeral=True
        )
        return

    if (
        chosen_model.startswith("meta-llama")
        or chosen_model.startswith("openrouter")
        or chosen_model.startswith("minimax")
        or chosen_model.startswith("google/")
        or chosen_model.startswith("nvidia/")
    ) and not OPENROUTER_API_KEY:
        await interaction.response.send_message(
            "OpenRouter API key is missing in environment variables.", ephemeral=True
        )
        return

    if tts_engine == "fish_audio" and not OPENROUTER_API_KEY:
        await interaction.response.send_message(
            "OpenRouter API key is missing in environment variables.", ephemeral=True
        )
        return

    await interaction.response.defer()

    app_info = await bot.application_info()
    is_owner = interaction.user.id == app_info.owner.id

    if is_owner:
        await interaction.edit_original_response(content="[Owner Bypass] Starting episode generation immediately...")
        asyncio.create_task(
            run_episode_job(interaction, topic, turns, chosen_model, tts_engine, include_user_chars, guild_id)
        )
        return

    queue_position = bot.episode_queue.qsize() + 1
    if queue_position > 1:
        await interaction.edit_original_response(
            content=f"Queued! Position in line: {queue_position - 1}"
        )
    else:
        await interaction.edit_original_response(content="Starting episode generation...")

    async def job(inter):
        await run_episode_job(inter, topic, turns, chosen_model, tts_engine, include_user_chars, guild_id)

    task_metadata = {
        "topic": topic,
        "user": interaction.user.display_name,
        "model": chosen_model,
        "turns": turns,
        "tts": tts_engine,
        "status": "Queued",
    }
    bot.queue_list.append(task_metadata)

    await bot.episode_queue.put((job, interaction, task_metadata))

@bot.tree.command(
    name="previewtext",
    description="Add preview text overlay to an image or animated GIF.",
)
@app_commands.describe(image="The base image or GIF you want to overlay onto")
@app_commands.choices(
    size=[
        app_commands.Choice(name="Full Overlay", value="full"),
        app_commands.Choice(name="150x150", value="150x150"),
    ]
)
async def previewtext(
    interaction: discord.Interaction,
    image: discord.Attachment,
    size: app_commands.Choice[str] = None,
):
    supported_types = ["image/png", "image/jpeg", "image/webp", "image/gif"]
    if image.content_type not in supported_types:
        await interaction.response.send_message(
            "Please upload a valid image (PNG, JPG, WEBP, or GIF).",
            ephemeral=True,
        )
        return

    if not os.path.exists(OVERLAY_FILENAME):
        await interaction.response.send_message(
            f"Server error: `{OVERLAY_FILENAME}` is missing.",
            ephemeral=True,
        )
        return

    await interaction.response.defer()

    size_mode = size.value if size else "full"

    try:
        user_image_bytes = await image.read()
        user_image_stream = io.BytesIO(user_image_bytes)

        with Image.open(user_image_stream) as base_img, Image.open(OVERLAY_FILENAME) as foreground:
            is_animated = getattr(base_img, "is_animated", False)
            output_buffer = io.BytesIO()

            bg_width, bg_height = base_img.size

            if size_mode == "150x150":
                target_w, target_h = 150, 150
            else:
                target_w, target_h = bg_width, bg_height

            foreground_rgba = foreground.convert("RGBA").resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

            if is_animated:
                processed_frames = []
                durations = []

                for frame in ImageSequence.Iterator(base_img):
                    frame_rgba = frame.convert("RGBA")
                    if (bg_width, bg_height) != (target_w, target_h):
                        frame_rgba = frame_rgba.resize(
                            (target_w, target_h), Image.Resampling.LANCZOS
                        )

                    frame_rgba.paste(foreground_rgba, (0, 0), mask=foreground_rgba)
                    processed_frames.append(
                        frame_rgba.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE)
                    )
                    durations.append(frame.info.get("duration", 100))

                loop = base_img.info.get("loop", 0)

                processed_frames[0].save(
                    output_buffer,
                    format="GIF",
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=durations,
                    loop=loop,
                    optimize=False,
                )
                output_filename = "preview_result.gif"
            else:
                background = base_img.convert("RGBA")
                if (bg_width, bg_height) != (target_w, target_h):
                    background = background.resize(
                        (target_w, target_h), Image.Resampling.LANCZOS
                    )

                background.paste(
                    foreground_rgba, (0, 0), mask=foreground_rgba
                )

                background.save(output_buffer, format="PNG")
                output_filename = "preview_result.png"

            output_buffer.seek(0)

        result_file = discord.File(output_buffer, filename=output_filename)
        await interaction.followup.send(file=result_file)

    except Exception as e:
        await interaction.followup.send(f"Failed to process image: {e}")


@bot.tree.command(name="version", description="Check current bot version status.")
async def version(interaction: discord.Interaction):
    image_url = "https://i.pinimg.com/736x/c9/f7/12/c9f712fe42b39c5651b214ca8efdc6a3.jpg"
    embed = discord.Embed(title="Running on LITE", color=discord.Color.blue())
    embed.set_image(url=image_url)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="version2", description="Information about version 2.")
async def version2(interaction: discord.Interaction):
    image_url = "https://i.pinimg.com/736x/c9/f7/12/c9f712fe42b39c5651b214ca8efdc6a3.jpg"
    embed = discord.Embed(title="Version 2 is coming soon", color=discord.Color.blue())
    embed.set_image(url=image_url)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="helpcommand", description="List available commands.")
async def help_command(interaction: discord.Interaction):
    help_text = (
        "**/episode [topic] [turns] [model] [tts]** - Generate an AI parody script (Queued)\n"
        "**/queue** - Check details of active/queued episode requests\n"
        "**/previewtext [image]** - Add preview text overlay to an image\n"
        "**/version** - Check bot version\n"
        "**/version2** - About version 2"
    )
    await interaction.response.send_message(help_text, ephemeral=True)


if __name__ == "__main__":
    if not DISCORD_TOKEN or not GEMINI_API_KEY:
        print("Error: DISCORD_TOKEN or GEMINI_API_KEY is missing from .env.")
    else:
        bot.run(DISCORD_TOKEN)
