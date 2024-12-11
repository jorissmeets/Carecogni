import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import datetime as dt

# Assign values for the types of surgery
operatie_adjustment = {
    "Heup": -0.1,
    "Knie": 0,
    "Femur": 0.1
}


# Sidebar input fields
st.sidebar.header("Input Fields")
leeftijd = st.sidebar.slider("Leeftijd van de patiënt", 18, 100, 25)
bmi = st.sidebar.slider("BMI van de patiënt", 15, 45, 25)
asa = st.sidebar.slider("ASA van de patiënt", 1, 5, 2)
dyspnoe_score = st.sidebar.slider("Dyspnoe score van de patiënt", 1, 5, 3)
ses_score = st.sidebar.slider("SES score van de patiënt", -100, 100, 0)
soort_operatie = st.sidebar.selectbox("Soort operatie", ["Knie", "Heup", "Femur"])

# Input data
input_data = [leeftijd, bmi, asa, dyspnoe_score, ses_score]

# Averages (replace with real values if available)
average_data = [50, 25, 3, 3, 0]

# Maximum values for each parameter
max_values = [100, 45, 5, 5, 200]  # SES normalized to run from 0 to 200

# Normalize SES score to run from 0 to 200
normalized_ses = ses_score*-1 + 100  # Shift range from -100 to 100 to 0 to 200
normalized_input = [
    value / max_val if param != "SES" else normalized_ses / max_values[-1]
    for value, max_val, param in zip(input_data, max_values, ["Leeftijd", "BMI", "ASA", "Dyspnoe", "SES"])
]

normalized_average = [
    value / max_val if param != "SES" else (average_data[-1] + 100) / max_values[-1]
    for value, max_val, param in zip(average_data, max_values, ["Leeftijd", "BMI", "ASA", "Dyspnoe", "SES"])
]

# Labels for the radar chart
parameters = ["Leeftijd", "BMI", "ASA", "Dyspnoe", "SES"]

# Radar chart function
def create_radar_chart(normalized_input, normalized_avg, labels, max_values):
    # Create the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Append the first value to close the radar chart
    normalized_input += normalized_input[:1]
    normalized_avg += normalized_avg[:1]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized_avg, color='black', linewidth=2, linestyle='-', label='Gemiddelde')
    ax.fill(angles, normalized_input, color='blue', alpha=0.5, label='Input')
    
    # Customize the chart
    ax.set_ylim(0, 1)  # Ensure all axes are scaled 0 to 1
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(0, 1, 10))
    ax.set_yticklabels([])  # Remove y-tick labels
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig

# Generate the radar chart
radar_chart = create_radar_chart(normalized_input, normalized_average, parameters, max_values)

# Display the radar chart in Streamlit
# st.title("Ligdagen Voorspelling")
# st.pyplot(radar_chart)

# Prediction Logic
prediction_score = np.mean(normalized_input)  # Simplified prediction logic based on mean
categories = ["Thuis", "Waarschijnlijk thuis", "Twijfelgeval", "Waarschijnlijk revalidatiecentrum", "Revalidatiecentrum"]
thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]

# Map the score to a category
if prediction_score < thresholds[0]:
    prediction = "Thuis"
elif prediction_score < thresholds[1]:
    prediction = "Waarschijnlijk thuis"
elif prediction_score < thresholds[2]:
    prediction = "Twijfelgeval"
elif prediction_score < thresholds[3]:
    prediction = "Waarschijnlijk revalidatiecentrum"
else:
    prediction = "Revalidatiecentrum"

# Prediction Logic
prediction_score = np.mean(normalized_input)  # Simplified prediction logic based on mean
categories = ["Thuis", "", "", "", "Revalidatiecentrum"]
thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]

def create_gauge_chart(score, thresholds, categories):
    fig = go.Figure()

    # Add gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, thresholds[0]], "color": "green"},
                {"range": [thresholds[0], thresholds[1]], "color": "lightgreen"},
                {"range": [thresholds[1], thresholds[2]], "color": "yellow"},
                {"range": [thresholds[2], thresholds[3]], "color": "orange"},
                {"range": [thresholds[3], thresholds[4]], "color": "red"},
            ],
            "bar": {"color": "blue"},
        },
        number={"suffix": ""},
        # title={"text": "Ontslaglocatie Score"},
        domain={"x": [0, 1], "y": [0, 1]}
    ))

    # Add labels for categories
    for i, (category, start, end) in enumerate(zip(categories, [0] + thresholds[:-1], thresholds)):
        if category == "":
            continue
        angle = np.pi * (1 - (start + end) / 2)  # Midpoint of the segment
        fig.add_annotation(
            x=0.5 + 0.48 * np.cos(angle),  # Adjusted x position (keep same radius)
            y=-0.1 + 0.7 * np.sin(angle),  # Lower the label by reducing the multiplier
            text=category,
            showarrow=False,
            font=dict(size=16, color="black", family="Arial"),  # Font styling
            align="center",
            bgcolor="white",  # Background color
            bordercolor="black",  # Border color
            borderwidth=1  # Border width
        )

    # Adjust margins to move the chart upwards
    fig.update_layout(
        margin=dict(t=10, b=0, l=0, r=0),  # Reduce top margin
        height=350  # Adjust height if needed
    )
    return fig

# Adjust prediction score based on the selected surgery type
adjusted_prediction_score = prediction_score + operatie_adjustment[soort_operatie]

# Ensure the adjusted score stays within the valid range (0 to 1)
adjusted_prediction_score = max(0, min(adjusted_prediction_score, 1))

# Update gauge chart generation to use the adjusted score
col1, col2 = st.columns(2)
gauge_chart = create_gauge_chart(adjusted_prediction_score, thresholds, categories)

# Place the radar chart in the first column
with col1:
    st.subheader("Vergelijking van Patiëntgegevens")
    st.pyplot(radar_chart)

# Place the gauge chart in the second column
with col2:
    st.subheader("Ontslaglocatie Voorspelling")
    st.plotly_chart(gauge_chart)


# Create the prediction when a patient will be discharged
def create_discharge_prediction_chart(adjusted_prediction_score):
    # Calculate the expected discharge date
    discharge_date = dt.date.today() + dt.timedelta(days=int(min(adjusted_prediction_score * 10-3,1)))
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    # Add a bar for the predicted discharge date
    fig.add_trace(go.Bar(
        x=[(discharge_date - dt.date.today()).days],  # Length of the bar in days
        y=["Ontslagdatum"],  # Label for the bar
        orientation='h',  # Horizontal bar
        marker=dict(color='red'),  # Bar color
        name="Voorspelde ontslagdatum"
    ))

    # Customize the layout
    fig.update_layout(
        xaxis=dict(
            title="Datum",
            tickvals=[i for i in range(0, 10, 1)],  # Tick marks every 5 days
            ticktext=[f"{(dt.date.today() + dt.timedelta(days=i)).strftime('%d-%m')}" for i in range(0, 10, 1)],
            showgrid=True
        ),
        yaxis=dict(title=""),
        height=300,
        bargap=0.2,
        showlegend=False  # Remove the legend for "Vandaag"
    )
    return fig

# Adjust prediction score based on user input
adjusted_prediction_score = prediction_score + operatie_adjustment[soort_operatie]
adjusted_prediction_score = max(0, min(adjusted_prediction_score, 1))

# Generate the prediction chart
prediction_chart = create_discharge_prediction_chart(adjusted_prediction_score)
# Calculate and format the predicted discharge date
predicted_discharge_date = dt.date.today() + dt.timedelta(days=int(max(adjusted_prediction_score * 10 - 3, 0)))
prediction_text = f"**Voorspelde ontslagdatum:** {predicted_discharge_date.strftime('%d-%m-%Y')}"

# Display the prediction in a panel
st.markdown(prediction_text)
# Display the chart