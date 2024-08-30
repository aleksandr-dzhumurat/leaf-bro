import streamlit as st
import requests

# Set the title of the app
st.title("Leaf Bro")

# Checkbox for relief options
st.header("Relief")
relief_options = ["Anxiety", "Depression", "Eye Pressure", "Insomnia", "Stress"]

# Arrange relief options in 3 columns
relief_cols = st.columns(3)
selected_relief = []
for i, option in enumerate(relief_options):
    with relief_cols[i % 3]:
        selected_relief.append(st.checkbox(option))

# Checkbox for positive effects
st.header("Positive Effects")
positive_effects_options = ["Euphoria", "Happy", "Hungry", "Uplifting"]

# Arrange positive effects options in 3 columns
effects_cols = st.columns(3)
selected_effects = []
for i, option in enumerate(positive_effects_options):
    with effects_cols[i % 3]:
        selected_effects.append(st.checkbox(option))

# Text input for search
st.header("Search")
search_query = st.text_input("Enter your search query:")

# Initialize session state to store results and checkboxes
if "results" not in st.session_state:
    st.session_state.results = []
if "checkboxes" not in st.session_state:
    st.session_state.checkboxes = []

# Place search button in a single column
if st.button("Search"):
    # Prepare data for the POST request
    data = {
        "relief": [relief_options[i] for i in range(len(relief_options)) if selected_relief[i]],
        "positive_effects": [positive_effects_options[i] for i in range(len(positive_effects_options)) if selected_effects[i]],
        "query": search_query
    }
    
    # Send POST request to the specified endpoint
    response = requests.post("http://0.0.0.0:8000/search", json=data)
    
    # Store results in session state
    st.session_state.results = response.json()
    st.session_state.checkboxes = [False] * len(st.session_state.results)

# Display the results in a table with checkboxes
if st.session_state.results:
    st.header("Search Results")
    st.write("Mark checkbox if result is relevant")
    
    for i, result in enumerate(st.session_state.results):
        col1, col2, col3 = st.columns([4, 6, 1])
        col1.write(result['title'])
        col2.write(result['explanation'])
        st.session_state.checkboxes[i] = col3.checkbox('', value=st.session_state.checkboxes[i], key=i)

    # Place "Score" and "Clear" buttons in a single row below the results
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Score"):
            # Prepare data for scoring
            score_data = [{"title": result['title'], "relevant": st.session_state.checkboxes[i]} 
                          for i, result in enumerate(st.session_state.results)]
            # Send POST request with scoring data
            score_response = requests.post("http://0.0.0.0:8000/score", json=score_data)
            # Display score response
            st.write("Score Response:", score_response.json())

    with col2:
        if st.button("Clear"):
            # Reset session state
            st.session_state.results = []
            st.session_state.checkboxes = []
            st.experimental_rerun()
