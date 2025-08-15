import os
import streamlit as st
from openai import OpenAI

# Create OpenAI client (expects OPENAI_API_KEY in environment or Streamlit Secrets)
client = OpenAI()

def estimate_max_tokens(target_words: int) -> int:
    # Roughly 1 token ≈ 0.75 words -> tokens ≈ words * 1.33
    tokens = int(target_words * 1.33)
    return max(256, min(tokens, 4096))

def build_user_prompt(
    premise: str,
    genre: str,
    tone: str,
    pov: str,
    audience: str,
    language: str,
    include_title: bool,
    include_dialogue: bool,
    themes: str,
    characters: str,
    setting: str,
    target_words: int,
) -> str:
    parts = []
    parts.append("Task: Write an original, engaging story that follows these specifications.")
    parts.append(f"Premise: {premise.strip() or 'Create a compelling story from scratch inspired by the genre and tone.'}")
    parts.append(f"Genre: {genre}")
    parts.append(f"Tone/Mood: {tone}")
    parts.append(f"Point of View: {pov}")
    parts.append(f"Intended Audience: {audience}")
    parts.append(f"Primary Language: {language}")
    parts.append(f"Target Length (approx.): {target_words} words")
    if themes.strip():
        parts.append(f"Key Themes and Motifs to Emphasize: {themes}")
    if characters.strip():
        parts.append(f"Characters (names, roles, traits): {characters}")
    if setting.strip():
        parts.append(f"Setting (time/place/atmosphere): {setting}")
    parts.append(f"Include a Title: {'Yes' if include_title else 'No'}")
    parts.append(f"Include Dialogue: {'Yes' if include_dialogue else 'No'}")
    parts.append("Requirements:")
    req = [
        "- Keep the narrative coherent with clear beginning, middle, and end.",
        "- Show, don't just tell. Use sensory details and vivid imagery.",
        "- Maintain consistent POV and tense, unless stylistically justified.",
        "- Avoid explicit content; keep it appropriate for the selected audience.",
        "- Use natural, character-revealing dialogue where relevant." if include_dialogue else "- Minimize dialogue; focus on narration and description.",
        "- If a title is requested, place it on the first line and then a blank line before the story." if include_title else "- Do not include a title line."
    ]
    parts.extend(req)
    return "\n".join(parts)

def generate_story(
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "You are an award-winning fiction writer who crafts vivid, emotionally resonant stories while strictly following the user's specifications."},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return response.choices[0].message.content

def app():
    st.set_page_config(page_title="AI Story Creator", page_icon="📖", layout="wide")
    st.title("📖 AI Story Creator")
    st.write("Generate a custom story by specifying your preferences. Set your OpenAI API key in the OPENAI_API_KEY environment variable.")

    with st.sidebar:
        st.header("Model & Generation Settings")
        model = st.selectbox("Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
        temperature = st.slider("Creativity (temperature)", 0.0, 2.0, 0.9, 0.1)
        default_words = 900
        target_words = st.slider("Target length (words)", min_value=200, max_value=4000, value=default_words, step=50)
        max_tokens = estimate_max_tokens(target_words)
        st.caption(f"Approx. max tokens: {max_tokens}")

    with st.form("story_form"):
        st.subheader("Story Specifications")
        premise = st.text_area("Premise or prompt", placeholder="E.g., A lighthouse keeper on a remote coast confronts a mysterious fog that carries whispers from the past.", height=100)
        col1, col2, col3 = st.columns(3)
        with col1:
            genre = st.selectbox("Genre", ["Fantasy", "Science Fiction", "Mystery/Thriller", "Horror", "Romance", "Historical Fiction", "Literary Fiction", "Adventure", "Young Adult", "Children's"], index=0)
        with col2:
            tone = st.selectbox("Tone", ["Whimsical", "Dark", "Uplifting", "Melancholic", "Hopeful", "Suspenseful", "Humorous", "Epic", "Cozy"], index=4)
        with col3:
            pov = st.selectbox("Point of view", ["First person", "Second person", "Third person limited", "Third person omniscient"], index=2)

        col4, col5, col6 = st.columns(3)
        with col4:
            audience = st.selectbox("Intended audience", ["Children", "Young Adult", "General", "Adult"], index=2)
        with col5:
            language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Hindi", "Japanese", "Korean", "Chinese"], index=0)
        with col6:
            include_title = st.checkbox("Include a title", value=True)

        include_dialogue = st.checkbox("Include dialogue", value=True)
        characters = st.text_input("Characters (optional)", placeholder="E.g., Mara (lighthouse keeper), Elias (fisherman), The Fog (ominous presence)")
        setting = st.text_input("Setting (optional)", placeholder="E.g., Storm-battered coast, 1890s; isolated lighthouse; rugged cliffs")
        themes = st.text_area("Themes & motifs (optional)", placeholder="E.g., memory, isolation vs. connection, nature’s indifference, redemption", height=80)

        submitted = st.form_submit_button("Generate Story", use_container_width=True)

    if submitted:
        with st.spinner("Crafting your story..."):
            try:
                user_prompt = build_user_prompt(
                    premise=premise,
                    genre=genre,
                    tone=tone,
                    pov=pov,
                    audience=audience,
                    language=language,
                    include_title=include_title,
                    include_dialogue=include_dialogue,
                    themes=themes,
                    characters=characters,
                    setting=setting,
                    target_words=target_words,
                )
                story = generate_story(
                    model=model,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                st.session_state["story_text"] = story
            except Exception as e:
                st.error(f"Failed to generate story: {e}")

    if "story_text" in st.session_state:
        st.subheader("Your Story")
        st.write(st.session_state["story_text"])

        file_name = "story.txt"
        st.download_button(
            label="Download Story",
            data=st.session_state["story_text"],
            file_name=file_name,
            mime="text/plain",
            use_container_width=True,
        )

if __name__ == "__main__":
    app()