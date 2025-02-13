import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import json
from datetime import datetime
from geopy.distance import geodesic
from difflib import SequenceMatcher
import branca.colormap as cm

def string_similarity(a, b):
    """Calculate string similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def load_data():
    # Load all datasets
    df_indicators = pd.read_csv('atendimento-ocorrencias/indicador_atendimento_filtrado.csv', 
                              sep=';', decimal=',', encoding='latin1')
    
    # Filter out unwanted indicators
    df_indicators = df_indicators[~df_indicators['SigIndicador'].isin(['NDIACRI', 'VLCLACRI'])]
    
    df_municipalities = pd.read_csv('indqual-municipio/indqual-municipio.csv',
                                  sep=';', decimal=',', encoding='latin1')
    
    # Load weather data and handle missing values
    df_weather = pd.read_csv('clima/clima.csv', sep=';', decimal=',', encoding='latin1')
    
    # Load municipalities coordinates
    with open('municipios.json', 'r', encoding='utf-8-sig') as f:
        municipalities_json = json.load(f)
    
    # Process weather data
    df_weather['Data Medicao'] = pd.to_datetime(df_weather['Data Medicao'], format='%Y-%m-%d')
    df_weather['Year'] = df_weather['Data Medicao'].dt.year
    df_weather['Month'] = df_weather['Data Medicao'].dt.month
    
    # Define available weather columns (excluding known missing ones)
    available_weather_columns = [
        'PRECIPITACAO TOTAL, HORARIO(mm)',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)',
        'UMIDADE RELATIVA DO AR, HORARIA(%)',
        'VENTO, VELOCIDADE HORARIA(m/s)'
    ]
    
    # Check which columns are actually available in the dataset
    weather_columns = [col for col in available_weather_columns if col in df_weather.columns]
    
    if not weather_columns:
        st.error("No weather measurement columns found in the data")
        return None, None, None, None
    
    # Add location and time columns
    weather_columns.extend(['Latitude', 'Longitude', 'Year', 'Month'])
    
    # Select only available columns
    df_weather = df_weather[weather_columns].copy()
    
    # Replace empty strings and invalid values with NaN
    df_weather = df_weather.replace(['', 'null', 'NA', 'NaN', '-'], np.nan)
    
    # Convert weather measurement columns to numeric, forcing errors to NaN
    for col in weather_columns:
        if col not in ['Latitude', 'Longitude', 'Year', 'Month']:
            try:
                df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert column {col} to numeric: {str(e)}")
                weather_columns.remove(col)
    
    # Convert Latitude and Longitude to numeric
    df_weather['Latitude'] = pd.to_numeric(df_weather['Latitude'], errors='coerce')
    df_weather['Longitude'] = pd.to_numeric(df_weather['Longitude'], errors='coerce')
    
    # Calculate monthly averages for weather data, excluding NaN values
    agg_dict = {col: 'mean' for col in weather_columns if col not in ['Latitude', 'Longitude', 'Year', 'Month']}
    # Use sum for precipitation instead of mean
    if 'PRECIPITACAO TOTAL, HORARIO(mm)' in agg_dict:
        agg_dict['PRECIPITACAO TOTAL, HORARIO(mm)'] = 'sum'
    
    try:
        df_weather_monthly = df_weather.groupby(['Year', 'Month', 'Latitude', 'Longitude']).agg(agg_dict).reset_index()
        # Convert all float columns to standard Python float type
        float_columns = df_weather_monthly.select_dtypes(include=['float64']).columns
        for col in float_columns:
            df_weather_monthly[col] = df_weather_monthly[col].astype(float)
    except Exception as e:
        st.error(f"Error in aggregating weather data: {str(e)}")
        return None, None, None, None
    
    # Remove rows where all weather measurements are NaN
    measurement_columns = [col for col in weather_columns if col not in ['Latitude', 'Longitude', 'Year', 'Month']]
    df_weather_monthly = df_weather_monthly.dropna(subset=measurement_columns, how='all')
    
    return df_indicators, df_municipalities, df_weather_monthly, municipalities_json

def string_similarity(a, b):
    """Calculate string similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def calculate_weather_correlations(df_indicators, df_weather, selected_conjunto, municipality_coords, selected_indicators):
    """Calculate correlations between selected indicators and weather data"""
    if municipality_coords is None or df_weather is None:
        return None
    
    # Find the nearest weather station
    nearest_station = find_nearest_weather_station(
        municipality_coords['latitude'], 
        municipality_coords['longitude'], 
        df_weather
    )
    if nearest_station is None:
        return None
    
    # Filter weather data for the nearest station
    station_weather = df_weather[
        (df_weather['Latitude'] == nearest_station[0]) & 
        (df_weather['Longitude'] == nearest_station[1])
    ]
    if station_weather.empty:
        st.warning("No weather data available for the nearest station")
        return None
    
    # Prepare the indicators data
    indicators_monthly = df_indicators[df_indicators['IdeConjUndConsumidoras'] == selected_conjunto].copy()
    indicators_monthly['Year'] = indicators_monthly['AnoIndice']
    indicators_monthly['Month'] = indicators_monthly['NumPeriodoIndice']
    
    # Merge weather and indicator data on Year and Month
    merged_data = pd.merge(
        indicators_monthly,
        station_weather,
        on=['Year', 'Month']
    )
    if merged_data.empty:
        st.warning("No matching time periods between weather and indicator data")
        return None
    
    # Get weather measurement columns (exclude location and time columns)
    weather_columns = [col for col in station_weather.columns 
                       if col not in ['Latitude', 'Longitude', 'Year', 'Month']]
    if not weather_columns:
        st.warning("No weather measurement columns available for correlation analysis")
        return None
    
    # Calculate correlations for each selected indicator
    correlations = {}
    for indicator in selected_indicators:
        indicator_data = merged_data[merged_data['SigIndicador'] == indicator]
        if indicator_data.empty:
            st.warning(f"No data available for indicator {indicator}")
            continue
        indicator_corr = {}
        for weather_col in weather_columns:
            # Compute correlation for each weather column individually
            corr_value = indicator_data['VlrIndiceEnviado'].corr(indicator_data[weather_col])
            indicator_corr[weather_col] = corr_value
        correlations[indicator] = indicator_corr
    
    return pd.DataFrame(correlations)  # rows will be weather columns, columns will be indicators

def find_nearest_weather_station(lat, lon, df_weather):
    """Find the nearest weather station based on coordinates"""
    min_distance = float('inf')
    nearest_coords = None
    
    station_coords = df_weather[['Latitude', 'Longitude']].drop_duplicates().values
    target_coords = (lat, lon)
    
    for station_lat, station_lon in station_coords:
        distance = geodesic(target_coords, (station_lat, station_lon)).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_coords = (station_lat, station_lon)
    
    return nearest_coords

def create_yearly_time_series(df, selected_conjunto, selected_indicators):
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Calculate yearly averages
    yearly_data = df.groupby(['AnoIndice', 'SigIndicador'])['VlrIndiceEnviado'].mean().reset_index()
    
    for indicator in selected_indicators:
        data = yearly_data[yearly_data['SigIndicador'] == indicator]
        
        fig.add_trace(go.Scatter(
            x=data['AnoIndice'],
            y=data['VlrIndiceEnviado'],
            name=indicator,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Yearly Average Values",
        xaxis_title="Year",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig

def create_monthly_time_series(df, selected_conjunto, selected_indicators, selected_year):
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Filter for selected year and calculate monthly averages
    monthly_data = df[df['AnoIndice'] == selected_year].groupby(
        ['NumPeriodoIndice', 'SigIndicador'])['VlrIndiceEnviado'].mean().reset_index()
    
    for indicator in selected_indicators:
        data = monthly_data[monthly_data['SigIndicador'] == indicator]
        
        fig.add_trace(go.Scatter(
            x=data['NumPeriodoIndice'],
            y=data['VlrIndiceEnviado'],
            name=indicator,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=f"Monthly Values for {selected_year}",
        xaxis_title="Month",
        yaxis_title="Value",
        xaxis=dict(tickmode='array', ticktext=[f'Month {i}' for i in range(1, 13)], tickvals=list(range(1, 13))),
        showlegend=True
    )
    
    return fig

def show_map_page(df_indicators, df_municipalities, municipalities_json):
    st.title("Mapa de Indicadores")
    
    unique_indicators = sorted(df_indicators['SigIndicador'].unique())
    selected_indicators_map = st.multiselect(
        "Select Indicators for the Map",
        unique_indicators,
        default=unique_indicators[:1]
    )
    if not selected_indicators_map:
        st.warning("Please select at least one indicator.")
        return
    
    latitudes = [item['latitude'] for item in municipalities_json if item.get('latitude') is not None]
    longitudes = [item['longitude'] for item in municipalities_json if item.get('longitude') is not None]
    if latitudes and longitudes:
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
    else:
        center_lat, center_lon = 0, 0
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    for indicator in selected_indicators_map:
        feature_group = folium.FeatureGroup(name=indicator)
        indicator_data = df_indicators[df_indicators['SigIndicador'] == indicator]
        aggregated = indicator_data.groupby('DscConjUndConsumidoras')['VlrIndiceEnviado'].mean().reset_index()
        if aggregated.empty:
            continue
        min_val = aggregated['VlrIndiceEnviado'].min()
        max_val = aggregated['VlrIndiceEnviado'].max()
        colormap = cm.linear.YlOrRd_09.scale(min_val, max_val)
        
        for _, row in aggregated.iterrows():
            muni_name = row['DscConjUndConsumidoras']
            value = row['VlrIndiceEnviado']
            coords = None
            for item in municipalities_json:
                if item.get('nome') == muni_name:
                    coords = (item.get('latitude'), item.get('longitude'))
                    break
            if coords:
                color = colormap(value)
                folium.CircleMarker(
                    location=coords,
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"{muni_name}: {value:.2f}"
                ).add_to(feature_group)
        feature_group.add_to(m)
    
    folium.LayerControl().add_to(m)
    folium_static(m)
    st.write("Color Scale Legend:")
    st.image(colormap._repr_html_())


def main():
    st.title("Ocorrências de Interrupções na Rede Elétrica na Região das Vertentes")

    
    try:
        # Load data
        df_indicators, df_municipalities, df_weather, municipalities_json = load_data()
        
        if df_weather is None:
            st.error("Não foi possível carregar os dados Climáticos")
            return

        
        # Navigation button: if "Mapa" is clicked, show the map page and stop further execution.
        if st.sidebar.button("Mapa"):
            show_map_page(df_indicators, df_municipalities, municipalities_json)
            st.stop()


        # Sidebar filters
        st.sidebar.header("Filtros")
        
        # Debug information
        st.sidebar.write("Numero total de dados:", len(df_indicators))
        
        # Create consumer unit options with municipality names
        conjunto_options = []
        unique_conjuntos = df_indicators['IdeConjUndConsumidoras'].unique()
        st.sidebar.write("Numero de Unidades Consumidoras:", len(unique_conjuntos))
        
        for conjunto in unique_conjuntos:
            # Safe indexing with error handling
            conjunto_data = df_indicators[df_indicators['IdeConjUndConsumidoras'] == conjunto]
            if not conjunto_data.empty and 'DscConjUndConsumidoras' in conjunto_data.columns:
                try:
                    municipality = conjunto_data['DscConjUndConsumidoras'].iloc[0]
                    conjunto_options.append(f"{conjunto} - {municipality}")
                except IndexError as e:
                    st.warning(f"Não foi possível pegar o nome do Munícipio do conjunto {conjunto}: {str(e)}")
                    conjunto_options.append(str(conjunto))
            else:
                conjunto_options.append(str(conjunto))
        
        # Filter by consumer unit
        if not conjunto_options:
            st.error("Sem unidades consumidoras disponíveis")
            return
            
        selected_conjunto_full = st.sidebar.selectbox("Selecione a Unidade Consumidora", conjunto_options)
        
        # Safely extract conjunto ID
        try:
            selected_conjunto = selected_conjunto_full.split(' - ')[0]
        except Exception as e:
            st.error(f"Erro ao processo o Conjunto selecionado: {str(e)}")
            return
        
        # Debug information
        st.sidebar.write("Conjunto selecionado:", selected_conjunto)
        
        # Filter by indicators
        indicators = sorted(df_indicators['SigIndicador'].unique())
        if not indicators:
            st.error("Não há indicadores selecionados")
            return
            
        selected_indicators = st.sidebar.multiselect(
            "Selecione os Indicadores",
            indicators,
            default=indicators[:3] if len(indicators) >= 3 else indicators
        )
        
        if not selected_indicators:
            st.warning("Selecione pelo menos um indicador")
            return
        
        # Year selection for monthly view
        years = sorted(df_indicators['AnoIndice'].unique())
        if not years:
            st.error("Não há anos disponíveis para o dado")
            return
            
        selected_year = st.sidebar.selectbox("Selecione o Ano", years, index=len(years)-1)
        
        # Time Series Analysis
        st.header("Análise Temporal")
        
        # Get filtered data for selected conjunto
        selected_conjunto = int(selected_conjunto)
        filtered_data = df_indicators[df_indicators['IdeConjUndConsumidoras'] == selected_conjunto]
        if filtered_data.empty:
            st.error(f"Não há dados disponiíveis para o Conjunto {selected_conjunto}")
        # Yearly analysis
        st.subheader("Anos (Média)")
        yearly_fig = create_yearly_time_series(
            filtered_data,
            selected_conjunto,
            selected_indicators
        )
        if yearly_fig:
            st.plotly_chart(yearly_fig)
        else:
            st.warning("Não foi possível criar a visualização dos anos")
        
        # Monthly analysis
        st.subheader("Meses")
        monthly_fig = create_monthly_time_series(
            filtered_data,
            selected_conjunto,
            selected_indicators,
            selected_year
        )
        if monthly_fig:
            st.plotly_chart(monthly_fig)
        else:
            st.warning("Não foi possivel criar a visualização dos meses")
        
        # Weather Correlation Analysis
        st.header("Correlação entre Indicadores X Climas")
        
        # Get municipality coordinates - with error handling
        try:
            conjunto_desc = filtered_data['DscConjUndConsumidoras'].iloc[0]
            st.write("Processando climas para o conjunto:", conjunto_desc)
        except IndexError:
            st.error("Erro ao buscar o nome do municipio para o conjunto")
            return
        except Exception as e:
            st.error(f"Erro ao buscar o nome do municipio: {str(e)}")
            return
        
        # Find matching municipality
        municipality_match = None
        best_ratio = 0
        for item in municipalities_json:
            try:
                ratio = string_similarity(item['nome'], conjunto_desc)
                if ratio > best_ratio and ratio > 0.6:
                    best_ratio = ratio
                    municipality_match = item
            except Exception as e:
                st.warning(f"Erro ao fazer o merge entre os nomes dos municipios {item.get('nome', 'unknown')}: {str(e)}")
                continue
        
        if municipality_match:
            st.write("Municipio encontrado:", municipality_match['nome'])
            
            
            correlations_df = calculate_weather_correlations(
                df_indicators,
                df_weather,
                selected_conjunto,
                municipality_match,
                selected_indicators  
            )

            if correlations_df is not None and not correlations_df.empty:
                for indicator in selected_indicators:
                    if indicator in correlations_df.columns:
                        # Create a one-row DataFrame for the current indicator
                        df_ind = pd.DataFrame(correlations_df[indicator]).T
                        fig = px.imshow(
                            df_ind,
                            text_auto=True,
                            labels=dict(x="Clima", y="Indicador", color="Correlação"),
                            color_continuous_scale="RdBu",
                            title=f"Correlation Heatmap for {indicator}"
                        )
                        st.plotly_chart(fig)
            else:
                st.warning("Não há correlações para mostrar")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data files and their format")
        # Print detailed error information
        import traceback
        st.error(f"Detailed error:\n{traceback.format_exc()}")
               
if __name__ == "__main__":
    main()
