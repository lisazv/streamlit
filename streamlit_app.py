#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import plotly.express as px


# In[21]:


df = pd.read_csv('Large_Passenger_Plane_Crashes_1933_to_2009.csv')
df.head()


# In[22]:


df.info()


# In[23]:


#Date omzetten naar DateTime

# Verwijder eventuele leading/trailing whitespaces van de 'Date' kolom
df['Date'] = df['Date'].str.strip()

# Converteer de 'Date' kolom naar een datetime object, negeer de tijd
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')

def correct_year(dt):
    if pd.isnull(dt):
        return dt  # Retourneer NaT zoals het is
    if dt.year > 2010:
        return dt.replace(year=dt.year-100)  # Aanpassen van het jaartal
    return dt  # Retourneer het oorspronkelijke datetime object als het jaartal correct is

# Corrigeer de jaartallen in de 'Datetime' kolom
df['Date'] = df['Date'].apply(correct_year)

df['Date'] = pd.to_datetime(df['Date'])


# In[5]:


# Zoek en vervang de foute waarneming
df['Time'] = df['Time'].str.replace('c14.30', '14:30')

# Converteer de 'Time' kolom naar een datetime datatype, maar behoud alleen het tijdstuk
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.time


# In[6]:


#som van missing values per kolom
df.isna().sum()


# In[7]:


#missing values droppen die alleen 5% van de data bevatten
threshold = len(df) * 0.05


# In[8]:


cols_to_drop = df.columns[df.isna().sum() <= threshold]

# Missing values droppen die onder de drempelwaarde zijn. 
df.dropna(subset=cols_to_drop, inplace=True)

df.isna().sum()


# In[9]:


#we droppen "Flight..", "cn.In" en "Route"
df = df.drop(columns=["Flight..", "cn.In", "Route"])


# In[10]:


df.isna().sum()


# In[44]:


#Manufacturer halen uit kollom 'Type'

def get_manufacturer(type_value):
    manufacturers = [
        ("Boeing", "Boeing"),
        ("Douglas", "Douglas"),
        ("Bristol", "Bristol"),
        ("Antonov", "Antonov"),
        ("Airbus", "Airbus"),
        ("Tupolev", "Tupolev"),
        ("Lockheed", "Lockheed"),
        ("Fokker", "Fokker"),
        ("Cesna", "Cesna"),
        ("Concorde", "Concorde")
    ]
    for keyword, manufacturer in manufacturers:
        if keyword in type_value:
            return manufacturer
    return "Other"

# Pas de functie get_manufacturer toe op de kolom 'Type'
df['Manufacturer'] = df['Type'].apply(lambda x: get_manufacturer(x) if isinstance(x, str) else "Other")


# In[12]:


import pandas as pd

# Oorspronkelijke code die werd gebruikt om landen te bepalen
Oorspronkelijke_code = """
import requests

def get_country_mapquest(location, api_key):
    # Set up the URL and parameters for the MapQuest Geocoding API
    url = "http://www.mapquestapi.com/geocoding/v1/address"
    params = {
        "key": api_key,
        "location": location,
        "outFormat": "json",
        "thumbMaps": False
    }
    
    try:
        # Make the request to the MapQuest Geocoding API
        response = requests.get(url, params=params)
        response.raise_for_status()  # Check for errors
        data = response.json()
        # Get the country from the geocoding result
        if data['results'] and data['results'][0]['locations']:
            country = data['results'][0]['locations'][0].get('adminArea1', 'Unknown')
            return country
        else:
            return 'Unknown'
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Unknown'

api_key = "cZXjjOWszFU4LMMOMLURNM9l1NkRCbz4"
df['Country'] = df['Location'].apply(lambda x: get_country_mapquest(x, api_key))
"""

# Laad de opgeslagen landgegevens
landen_df = pd.read_csv('landen.csv')

# Voeg de opgeslagen landgegevens samen met uw oorspronkelijke dataset
df = df.merge(landen_df, on='Location', how='left')


# In[13]:


# Lees de landen dataset
countries_df = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')

# Selecteer de relevante kolommen uit de landen dataset
selected_countries_df = countries_df[['name', 'region', 'alpha-2']]

# Combineer de datasets op basis van de landafkortingen
df = df.merge(selected_countries_df, left_on='Country', right_on='alpha-2', how='left')

# Verwijder de kolom 'alpha-2' uit de samengevoegde DataFrame
df = df.drop(columns='alpha-2')

# Hernoem de kolomnamen
df.rename(columns={'name': 'Country Name', 'region': 'Region'}, inplace=True)


# In[25]:


# Oorspronkelijke code die werd gebruikt om during en cause te bepalen
Oorspronkelijke_code2 = """
pip install openai

import openai
import pandas as pd

# Stel API-sleutel in
openai.api_key = "Vul hier API-key in"


# Definieer functie om te achterhalen tijdens welke fase de crash heeft plaatsgevonden
def get_when(summary):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=(f"When did the crash occur? (Select only 1: Pre-flight, Pushback, Taxi, Takeoff, Climb, Cruise, Descent, Approach, Landing, Taxi to Gate, Other, Unknown)(If not sure always choose Unknown): '{summary}'\n\nAnswer: "),
        temperature=0,
        max_tokens=10
    )
    cause = response.choices[0].text.strip()
    return cause

# Pas deze functie toe op de 'Summary' kolom van je DataFrame
df['During'] = df['Summary'].apply(get_when)



# Stel API-sleutel in
openai.api_key = "Vul hier API-key in"


# Definieer een functie om de oorzaak van een ongeluk op te halen
def get_cause(summary):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=(f"What caused the plane crash? (Select only 1: Human Factors, Technical Failures, Weather Conditions, Bird Strikes, Sabotage/Terrorism, Fires/Explosions, Air Traffic Errors, Runway Incursions, Unknown, Other): '{summary}'\n\nAnswer: "),
        temperature=0,
        max_tokens=10
    )
    cause = response.choices[0].text.strip()
    return cause

# Pas deze functie toe op de 'Summary' kolom van je DataFrame
df['Cause'] = df['Summary'].apply(get_cause)
"""

# Laad de opgeslagen kollomen "During" en "Cause"
During_Cause_df = pd.read_csv('during_cause.csv')

# Voeg de opgeslagen kollomen samen met oorspronkelijke dataset
df = df.merge(During_Cause_df, on='Summary', how='left')


# In[26]:


#Mask om te lange antwoorden chat-gpt eruit te halen:
mask = df['During'].str.len() <= 13
df = df[mask]


# In[16]:


#Military or Non-Military kollom toevoegen
df["M_nonM"] = "Non-Military"
for index, row in df.iterrows():
    if "Military" in row["Operator"]:
        df.at[index, "M_nonM"] = "Military"


# In[27]:


df.head()


# In[57]:
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year  # Voeg een jaar-kolom toe



# Dropdown menu voor het selecteren van een categorie
st.sidebar.title("Interactieve Analyse van Vliegtuigongevallen")
st.sidebar.write('Er wordt onderzoek gedaan naar vliegtuigongevallen tussen 1933 en 2009. \
We kijken naar militaire vliegtuigongevallen,naar de type fabrikant, welke vluchtfase \
 er een vliegtuigongeval plaatsvond en naar de oorzaak.')

# Slider voor het filteren op jaar
min_year, max_year = st.sidebar.slider(
    'Selecteer een jaarrange',
    min_value=df['Date'].dt.year.min(),
    max_value=df['Date'].dt.year.max(),
    value=(df['Date'].dt.year.min(), df['Date'].dt.year.max())
)

unique_regions = df['Region'].unique().tolist()

# Voeg een optie toe om alle regio's te selecteren
options = ['Alle regio’s'] + unique_regions

# Dropdown selectie box
selected_region = st.sidebar.selectbox(
    'Selecteer een regio:',
    options=options,
    index=0  # Standaardwaarde is 'Alle regio’s'
)

# Als 'Alle regio’s' is geselecteerd, gebruik dan alle unieke regio's. Anders, gebruik de geselecteerde regio.
if selected_region == 'Alle regio’s':
    selected_regions = unique_regions
else:
    selected_regions = [selected_region]

# Filter de DataFrame op basis van de geselecteerde jaren en regio's
filtered_df = df[
    (df['Date'].dt.year.between(min_year, max_year)) &
    (df['Region'].isin(selected_regions))
]

df['Date'] = pd.to_datetime(df['Date'])  # Converteer de datumkolom naar datetime

selected_category = st.sidebar.selectbox('Selecteer een categorie', options=['Militaire vluchten', 'Fabrikant', 'Vlucht Fase', 'Oorzaak'])

# Functie om een plot te genereren op basis van de geselecteerde categorie
def generate_plot(category):
    if category == 'Vlucht Fase':
        st.subheader("Relatie tussen de overlevingskans en vluchtfase")
        st.write("Hier wordt er gekeken naar de overlevingskansen voor verschillende vluchtfases waar er een ongeluk plaatsvond.")
        fig = px.box(filtered_df, x='SurvivalRate', y='During', color='During', 
        title='Overlevingskans per vluchtfase',
        labels={'SurvivalRate': 'Overlevingskans', 'During': 'Vluchtfase'})
        
        fig.update_traces(marker=dict(line=dict(width=2)))
        fig.update_layout(showlegend=False)
        
        st.plotly_chart(fig)
        
    elif category == 'Militaire vluchten':
        st.subheader("Relatie tussen Militair en niet-Militaire vliegtuigongenlukken")
        st.write("Hier wordt er gekeken naar de aantal overledenen per jaar per vluchtfase. Dit hebben we verdeeld tussen militaire en niet-militaire vluchten.")
        fig = px.scatter(data_frame=filtered_df, y= "Fatalities", x = "Date", color = "During")
        unique_M_nonM = df['M_nonM'].unique()
        
        # Create a dictionary to store button configurations
        buttons = [{'label': M_nonM, 'method': 'update', 'args': [{'visible': [M_nonM == r for r in df['M_nonM']]}]}
        for M_nonM in unique_M_nonM]

        # Add buttons to the plot and show
        fig.update_layout(
        title='Dodelijke ongevallen over tijd per vluchtfase (Militair vs Niet-militair)',
        xaxis_title='Datum',
        yaxis_title='Aantal Dodelijke Slachtoffers',
        updatemenus=[{
        'type': "dropdown", 'direction': 'down',
        'x': 1.3, 'y': 0.2,
        'showactive': True, 'active': 0,
        'buttons': buttons
        }]
        )

        st.plotly_chart(fig)

    elif category == 'Oorzaak':
        st.subheader("Relatie tussen de oorzaak en overlevingskans")
        st.write("Er wordt een boxplot weergegeven waarbij de distributie van de overlevingskansen te zien is tegenover de oorzaak.")
        fig = px.box(filtered_df, x='SurvivalRate', y='Cause', color='Cause', 
        title='Overlevingskans per oorzaak',
        labels={'SurvivalRate': 'Overlevingskans', 'Cause': 'Oorzaak'})
        
        fig.update_traces(marker=dict(line=dict(width=2)))
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig)

    elif category == 'Fabrikant':
        st.subheader("Relatie tussen de Fabrikant van het vliegtuig en het aantal ongevallen.")
        st.write("Om de relatie te zien tussen het fabrikant en het aantal ongevallen, wordt er een boxplot gemaakt om de distributie weer te geven per fabrikant")
        fig = px.histogram(filtered_df, x="Year", color="Manufacturer", opacity = 0.7, barmode ="group")
        fig.update_layout(
        title='Aantal vliegtuigongevallen per jaar per fabrikant',
        xaxis_title='Jaar',
        yaxis_title='Aantal ongevallen'
        )

        st.plotly_chart(fig)
        
    else:
        st.write(f'Geen plot beschikbaar voor {category}')

# Aanroep van de functie om de plot te genereren
generate_plot(selected_category)


# In[ ]:




