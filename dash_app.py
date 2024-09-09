import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
import plotly.express as px
from streamlit_elements import elements, mui, html, dashboard
import plotly.io as pio
import numpy as np
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
import time
from streamlit_folium import folium_static
import random 
import requests
import math


# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

@st.cache_data
def load_data(file, user_type=None):
    """Loads data and filters by user type if specified."""
    if file is not None:
        # Read the file as a buffer
        file.seek(0)  # Ensure the file pointer is at the start
        df = pd.read_csv(file)
        return df if user_type is None else df[df['usertype'] == user_type]
    return None


# Custom CSS to make tab text bigger
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
data_all = load_data(uploaded_file)
data_customer = load_data(uploaded_file, 'Customer')
data_subscriber = load_data(uploaded_file, 'Subscriber')


# Convert 'starttime' column to datetime
data_all['starttime'] = pd.to_datetime(data_all['starttime'], format='%m/%d/%Y %H:%M')
data_customer['starttime'] = pd.to_datetime(data_customer['starttime'], format='%m/%d/%Y %H:%M')
data_subscriber['starttime'] = pd.to_datetime(data_subscriber['starttime'], format='%m/%d/%Y %H:%M')

# Get the minimum and maximum date from 'starttime'
min_date = data_all['starttime'].min().date()
max_date = data_all['starttime'].max().date()


# Sidebar
with st.sidebar:
    st.header("Date Range Selector")
    date_range = st.date_input(
        "Select a date range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range_picker"
    )

# Convert date_range to datetime for comparison
start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[1])

# Convert 'starttime' column to datetime and extract only the date part
data_all['starttime_date'] = data_all['starttime'].dt.date
data_customer['starttime_date'] = data_customer['starttime'].dt.date
data_subscriber['starttime_date'] = data_subscriber['starttime'].dt.date

# Filter the dataframe by the selected date range
data_all = data_all[(data_all['starttime_date'] >= start_date.date()) & (data_all['starttime_date'] <= end_date.date())]
data_customer = data_customer[(data_customer['starttime_date'] >= start_date.date()) & (data_customer['starttime_date'] <= end_date.date())]
data_subscriber = data_subscriber[(data_subscriber['starttime_date'] >= start_date.date()) & (data_subscriber['starttime_date'] <= end_date.date())]

def display_centered_text(text, size='24px'):
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 50px;">
            <h1 style="font-size: {size};">{text}</h1>
        </div>
    """, unsafe_allow_html=True)


def display_pie_chart(labels, sizes):
    """Displays a Pie chart using Plotly."""
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])
    fig.update_layout(
        width=600,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def categorize_age(age):
    if 16 <= age <= 20:
        return '16-20'
    elif 21 <= age <= 25:
        return '21-25'
    elif 26 <= age <= 35:
        return '26-35'
    elif 36 <= age <= 45:
        return '36-45'
    elif 46 <= age <= 55:
        return '46-55'
    else:
        return '56+'

# Prepare Subscriber Data
def prepare_subscriber_data(df):
    df['age'] = 2014 - df['birthyear']
    df['age_group'] = df['age'].apply(categorize_age)
    return df

sub_all = prepare_subscriber_data(data_subscriber)
sub_male = prepare_subscriber_data(data_subscriber[data_subscriber['gender'] == 'Male'])
sub_female = prepare_subscriber_data(data_subscriber[data_subscriber['gender'] == 'Female'])

# Aggregate Trip Counts
def aggregate_trip_counts(df):
    return df.groupby('age_group').size().reset_index(name='trip_count')

trip_male_counts = aggregate_trip_counts(sub_male)
trip_female_counts = aggregate_trip_counts(sub_female)
trip_all_counts = aggregate_trip_counts(sub_all)

trip_male_counts['gender'] = 'Male'
trip_female_counts['gender'] = 'Female'
trip_all_counts['gender'] = 'All'

trip_counts_combined = pd.concat([trip_all_counts, trip_male_counts, trip_female_counts])

def generate_hour_brackets():
    return [datetime.time(i) for i in range(24)]  # Generate time objects for each hour

def count_time_values_within_brackets(start_times, time_brackets):
    hour_counts = [0] * (len(time_brackets) - 1)  # Initialize counts

    for t in start_times:
        # Extract hour and minute from datetime
        hour = t.hour
        minutes = t.minute

        # Create a time object for comparison
        current_time = datetime.time(hour, minutes)

        # Check which bracket the current_time falls into
        for b in range(len(time_brackets) - 1):
            if time_brackets[b] <= current_time < time_brackets[b + 1]:
                hour_counts[b] += 1  # Increment the count for that hour bracket
                break  # Exit the loop once found

    return hour_counts

time_brackets = generate_hour_brackets()

time_bracket_list = [f"{time_brackets[i]} - {time_brackets[i + 1]}" for i in range(len(time_brackets) - 1)]

hour_counts_all = count_time_values_within_brackets(data_all['starttime'], time_brackets)
hour_counts_sub = count_time_values_within_brackets(sub_all['starttime'], time_brackets)
hour_counts_sub_male = count_time_values_within_brackets(sub_male['starttime'], time_brackets)
hour_counts_sub_female = count_time_values_within_brackets(sub_female['starttime'], time_brackets)
hour_counts_cus = count_time_values_within_brackets(data_customer['starttime'], time_brackets)

# Function to display the graph
def display_graph(hour_counts, title):
    fig = go.Figure(data=[go.Bar(
        x=time_bracket_list,
        y=hour_counts,
        marker_color='skyblue'
    )])

    fig.update_layout(
        title=title,
        xaxis_title='Hour Bracket',
        yaxis_title='Number of Occurrences',
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
        
        
        showlegend=False
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    st.plotly_chart(fig)

def display_clustered_graph(hour_counts_male, hour_counts_female, title):
    fig = go.Figure(data=[
        go.Bar(
            name='Male',
            x=time_bracket_list,
            y=hour_counts_male,
            marker_color='skyblue'
        ),
        go.Bar(
            name='Female',
            x=time_bracket_list,
            y=hour_counts_female,
            marker_color='lightcoral'
        )
    ])

    # Change the bar mode to 'group' for clustered bars
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title='Time Bracket',
        yaxis_title='Occurrences',
        legend_title='Gender',
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    st.plotly_chart(fig)

def count_trips_by_day_of_week(start_times):
    """Count the number of trips that start on each day of the week."""
    day_counts = start_times.dt.dayofweek.value_counts().reindex(range(7), fill_value=0).sort_index()
    return day_counts

# Function to display the graph
def display_graph_week(day_counts, title):
    fig = go.Figure(data=[go.Bar(
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        y=day_counts,
        marker_color='skyblue'
    )])

    fig.update_layout(
        title=title,
        xaxis_title='Days',
        yaxis_title='Number of Occurrences',
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
        
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    st.plotly_chart(fig)


day_count_all = count_trips_by_day_of_week(data_all['starttime'])

day_count_sub = count_trips_by_day_of_week(sub_all['starttime'])
day_count_sub_male = count_trips_by_day_of_week(sub_male['starttime'])
day_count_sub_female = count_trips_by_day_of_week(sub_female['starttime'])
day_count_cus = count_trips_by_day_of_week(data_customer['starttime'])


def display_clustered_graph_day(day_counts_male, day_counts_female, title):
    fig = go.Figure(data=[
        go.Bar(
            name='Male',
            x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            y=day_counts_male,
            marker_color='skyblue'
        ),
        go.Bar(
            name='Female',
            x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            y=day_counts_female,
            marker_color='lightcoral'
        )
    ])
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title='Day',
        yaxis_title='Occurrences',
        legend_title='Gender',
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    st.plotly_chart(fig)

def count_trips_by_month(start_times):
    # Convert the start times to datetime format using pandas
    start_times = pd.to_datetime(start_times, errors='coerce')
    
    # Filter for the first six months only
    start_times = start_times[start_times.dt.month <= 6]
    
    # Count trips per month
    month_counts = start_times.dt.month.value_counts().reindex(range(1, 7), fill_value=0).sort_index()
    
    return month_counts



# Count occurrences of trips by month
month_counts_all = count_trips_by_month(data_all['starttime'])
month_counts_sub = count_trips_by_month(sub_all['starttime'])
month_counts_cus = count_trips_by_month(data_customer['starttime'])
month_counts_sub_male = count_trips_by_month(sub_male['starttime'])
month_counts_sub_female = count_trips_by_month(sub_female['starttime'])


# Function to display the graph
def display_graph_month(month_counts, title):
    fig = go.Figure(data=[go.Bar(
        x=['January', 'February', 'March', 'April', 'May', 'June'],
        y=month_counts,
        marker_color='skyblue'
    )])

    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Number of Occurrences',
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    st.plotly_chart(fig)

def display_clustered_graph_month(month_counts_male, month_counts_female, title):
    fig = go.Figure(data=[
        go.Bar(
            name='Male',
            x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            y=month_counts_male,
            marker_color='skyblue'
        ),
        go.Bar(
            name='Female',
            x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            y=month_counts_female,
            marker_color='lightcoral'
        )
    ])
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title='Day',
        yaxis_title='Occurrences',
        legend_title='Gender',
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    st.plotly_chart(fig)


# Function to geocode address
@st.cache_data
def get_coordinates(address):
    geolocator = Nominatim(user_agent="app")
    try:
        location = geolocator.geocode(address)
        return [location.latitude, location.longitude] if location else None
    except Exception as e:
        st.error(f"Error geocoding address '{address}': {e}")
        return None

# Batch processing with progress display
@st.cache_data
def process_in_batches(data, batch_size, delay=4):
    coordinates = []
    progress_bar = st.progress(0)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        coordinates.extend([get_coordinates(address) for address in batch])
        progress_bar.progress(min((i + batch_size) / len(data), 1))
        time.sleep(delay)
    progress_bar.empty()
    return coordinates

# Function to fetch shortest route between two points using OSRM
@st.cache_data
def get_shortest_route(start, end):
    url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
    response = requests.get(url)
    if response.status_code == 200:
        route = response.json()['routes'][0]['geometry']['coordinates']
        return [[coord[1], coord[0]] for coord in route]  # Swap lat/lon for folium
    else:
        st.error(f"Error fetching route: {response.status_code}")
        return None

# Random color generator
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Haversine formula for distance calculation
def haversine(coord1, coord2):
    R = 6371  # Earth's radius in km
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Data processing function for visualization
@st.cache_data
def prepare_map_data(data, chicago_center, max_distance=100):
    map_data, route_data, route_names = [], [], []
    start_coords, end_coords, counts = data['start'], data['end'], data['count']
    
    for start, end, count, start_name, end_name in zip(start_coords, end_coords, counts, 
                                                       data['from_station_name'], data['to_station_name']):
        if start and end and haversine(chicago_center, start) <= max_distance and haversine(chicago_center, end) <= max_distance:
            map_data.extend([
                {'lat': start[0], 'lon': start[1], 'station': start_name, 'type': 'Start', 'count': count},
                {'lat': end[0], 'lon': end[1], 'station': end_name, 'type': 'End', 'count': count}
            ])
            route_coords = get_shortest_route(start, end)
            if route_coords:
                route_data.append(route_coords)
                route_names.append(f"{start_name}: {end_name}")
    
    return pd.DataFrame(map_data), route_data, route_names

# Reusable plotting function
def plot_routes(fig, route_data, route_names, counts):
    for route, route_name, count in zip(route_data, route_names, counts):
        if route:
            trace_name = f"{route_name[:6]}...{route_name[-10:]}"
            fig.add_trace(
                go.Scattermapbox(
                    lat=[point[0] for point in route],
                    lon=[point[1] for point in route],
                    mode="lines",
                    line=dict(width=4, color=generate_random_color()),
                    hovertext=f"{route_name}<br>counts: {count}",
                    hoverinfo='text',
                    visible='legendonly',
                    name=trace_name
                )
            )

# Streamlit Map visualization function
def render_tab(tab, data, chicago_center):
    with tab:
        col1, col2 = st.columns(2)
        with col2:
            st.write("See on Map")
            top_20_routes = data.groupby(['from_station_name', 'to_station_name']).size().reset_index(name='count')
            top_20_routes = top_20_routes.sort_values(by='count', ascending=False).head(20)

            start_stations = top_20_routes['from_station_name'].tolist()
            end_stations = top_20_routes['to_station_name'].tolist()

            with st.spinner("Geocoding stations..."):
                start_coords = process_in_batches(start_stations, batch_size=10)
                end_coords = process_in_batches(end_stations, batch_size=10)

            top_20_routes['start'], top_20_routes['end'] = start_coords, end_coords
            df, route_data, route_names = prepare_map_data(top_20_routes, chicago_center)

            fig = go.Figure()
            if route_data and route_names:
                plot_routes(fig, route_data, route_names, top_20_routes['count'])
                fig.update_layout(
                    mapbox=dict(style="carto-positron", zoom=11, center=dict(lat=chicago_center[0], lon=chicago_center[1])),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
            else:
                st.write("No routes available to display.")

            st.plotly_chart(fig)

        with col1:
            st.write("Top 20 Popular Routes")
            st.dataframe(top_20_routes[["from_station_name", "to_station_name"]], use_container_width=True)




st.title("Bike-Share Dashboard")
st.markdown("---")



# Key Metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trips", f"{len(data_all):,}")
col2.metric("Subscriber Trips", f"{len(data_subscriber):,}")
col3.metric("Customer Trips", f"{len(data_customer):,}")
col4.metric("Unique Stations", f"{data_all['from_station_id'].nunique():,}")



st.markdown("---")




# User Type Distribution and Demographics
st.header("Subscriber Demographics")
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Subscriber Gender Distribution")
    display_pie_chart(labels=['Male Trips', 'Female Trips'], sizes=[len(sub_male), len(sub_female)])

with col2:
    st.subheader("Subscriber Age Group Distribution")
    fig = px.bar(trip_counts_combined, 
                x='age_group', 
                y='trip_count', 
                color='gender', 
                barmode='group',
                labels={'trip_count': 'Trip Count', 'age_group': 'Age Group'})
                
    st.plotly_chart(fig)

st.markdown("---")


st.subheader("Average Trip Time (min)")

# Average Trip Times
col1, col2, col3 = st.columns(3)
col1.metric("All Users", f"{np.round(np.average(data_all['tripduration']) / 60, 1):,}")
col2.metric("Subscriber", f"{np.round(np.average(data_subscriber['tripduration']) / 60, 1):,}")
col3.metric("Customer", f"{np.round(np.average(data_customer['tripduration']) / 60, 1):,}")





with col2:
    with st.expander("Gender Breakdown"):
        st.metric("Male", f"{np.round(np.average(sub_male['tripduration']) / 60, 1):,}")
        st.metric("Female", f"{np.round(np.average(sub_female['tripduration']) / 60, 1):,}")

st.subheader("Trips Trends")


tab11, tab12, tab13 = st.tabs(["Hourly", "Weekly", "Monthly"])
# Trip Start Times
with tab11:
    st.subheader("Customer Type")
    tab1, tab2, tab3 = st.tabs(["All Users", "Subscribers", "Customers"])

    with tab1:
        display_graph(hour_counts_all, 'Hourly Trend (All Users)')

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            display_graph(hour_counts_sub, 'Hourly Trend (Subscribers)')
        with col2:
            display_clustered_graph(
                hour_counts_sub_male,
                hour_counts_sub_female,
                'Hourly Trend by Gender (Subscribers)'
            )

    with tab3:
        display_graph(hour_counts_cus, 'Hourly Trend (Customers)')

with tab12:
    # Trips by Day of Week
    st.subheader("Customer Type")
    tab1, tab2, tab3 = st.tabs(["All Users", "Subscribers", "Customers"])

    with tab1:
        display_graph_week(day_count_all, 'Weekly Trend (All Users)')

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            display_graph_week(day_count_sub, 'Weekly Trend (Subscribers)')
        with col2:
            display_clustered_graph_day(
                day_count_sub_male,
                day_count_sub_female,
                'Weekly Trend by Gender (Subscribers)'
            )

    with tab3:
        display_graph_week(day_count_cus, 'Trips by Day of Week (Customers)')


with tab13:
        
    # Trips by Day of Week
    st.subheader("Customer Type")
    tab1, tab2, tab3 = st.tabs(["All Users", "Subscribers", "Customers"])

    with tab1:
        display_graph_month(month_counts_all, 'Monthly Trend (All Users)')

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            display_graph_month(month_counts_sub, 'Monthly Trend (Subscribers)')
        with col2:
            display_clustered_graph_month(
                month_counts_sub_male,
                month_counts_sub_female,
                'Monthly Trend by Gender (Subscribers)'
            )

    with tab3:
        display_graph_month(month_counts_all, 'Monthly Trend (Customers)')


st.markdown("---")



# Main section
st.subheader("20 Most Popular Routes")
chicago_center = [41.8781, -87.6298]

# Tabs for different user types
tab_all, tab_subscribers, tab_customers = st.tabs(["All", "Subscribers", "Customers"])
render_tab(tab_all, data_all, chicago_center)
render_tab(tab_subscribers, data_subscriber, chicago_center)
render_tab(tab_customers,data_customer, chicago_center)
