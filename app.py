import gradio as gr
import pandas as pd
import joblib

# --- 1. LOAD THE TRAINED MODEL ---
MODEL_PATH = 'movie_revenue_predictor.pkl'
try:
    loaded_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    # This provides a helpful error if the model file is missing.
    print(f"Error: Model file not found at {MODEL_PATH}")
    loaded_model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    loaded_model = None

# --- 2. DEFINE THE FEATURE LISTS 

GENRE_CHOICES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
    'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

ACTOR_CHOICES = [
    'Adam Sandler', 'Arnold Schwarzenegger', 'Ben Stiller', 'Bruce Willis', 'Denzel Washington',
    'George Clooney', 'Harrison Ford', 'Johnny Depp', 'Matt Damon', 'Nicolas Cage', 'Other',
    'Robert De Niro', 'Sylvester Stallone', 'Tom Cruise', 'Tom Hanks'
]


DIRECTOR_CHOICES = [
    'Clint Eastwood', 'Gore Verbinski', 'James Cameron', 'Martin Scorsese', 'Michael Bay',
    'Other', 'Peter Jackson', 'Ridley Scott', 'Robert Zemeckis', 'Ron Howard',
    'Sam Raimi', 'Steven Soderbergh', 'Steven Spielberg', 'Tim Burton', 'Woody Allen'
]

MODEL_COLUMNS = ['budget', 'popularity', 'runtime', 'release_year'] + \
                [f'genre_{g}' for g in GENRE_CHOICES] + \
                [f'actor_{a}' for a in ACTOR_CHOICES] + \
                [f'director_{d}' for d in DIRECTOR_CHOICES]


# --- 3. CREATE THE PREDICTION FUNCTION ---
def predict_revenue(budget, popularity, runtime, release_year, main_genre, lead_actor, director):
    # Check if the model loaded successfully
    if loaded_model is None:
        return "Error: Model is not loaded. Please check the application logs."

    # Check for empty inputs
    if not all([budget, popularity, runtime, release_year, main_genre, lead_actor, director]):
        return "Error: Please fill in all fields before submitting."

    # Create a dictionary for the input data
    input_data = {
        'budget': budget, 'popularity': popularity,
        'runtime': runtime, 'release_year': release_year
    }
    input_data['genre_' + main_genre] = 1
    input_data['actor_' + lead_actor] = 1
    input_data['director_' + director] = 1

    # Convert to a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Align columns with the model's training data, filling missing with 0
    model_input = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    
    # Make a prediction
    prediction = loaded_model.predict(model_input)
    
    # Format the output
    return f"Predicted Worldwide Revenue: ${prediction[0]:,.0f}"


# --- 4. BUILD THE GRADIO INTERFACE ---
inputs = [
    gr.Number(label="Budget (in USD)", value=100000000),
    gr.Number(label="Popularity Score", value=150.0),
    gr.Number(label="Runtime (in minutes)", value=120),
    gr.Slider(minimum=1980, maximum=2025, step=1, label="Release Year", value=2015),
    gr.Dropdown(choices=sorted(GENRE_CHOICES), label="Main Genre", value="Action"),
    gr.Dropdown(choices=sorted(ACTOR_CHOICES), label="Lead Actor", value="Tom Hanks"),
    gr.Dropdown(choices=sorted(DIRECTOR_CHOICES), label="Director", value="Steven Spielberg")
]

output = gr.Textbox(label="Prediction")

app = gr.Interface(
    fn=predict_revenue,
    inputs=inputs,
    outputs=output,
    title="🎬 Movie Revenue Predictor",
    description="An end-to-end machine learning project to predict the worldwide box office revenue of a movie. Enter the details and click 'Submit' to see the prediction.",
    allow_flagging="never"
)

# Launch the app
app.launch()
