#--------------------------------------------------API
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import locale
import uvicorn
#--------------------------------------------------ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df_completo=pd.read_csv(r'Datasets/Data_ETL_EDA.csv')

app = FastAPI()

#http://127.0.0.1:8000

@app.get("/")
def index():
    return {"mensaje" : "Hola, bienvenidos!!"}

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    fechas=pd.to_datetime(df_completo['release_date'],format='%Y-%m-%d')
    nmes=fechas[fechas.dt.month_name(locale='es_CO')==mes.capitalize()]
    respuesta=nmes.shape[0]
    return {'mes':mes, 'cantidad':respuesta}

@app.get('/cantidad_filmaciones_dia{dia}')
def cantidad_filmaciones_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8') #configuración regional
    fechas=pd.to_datetime(df_completo['release_date'],format='%Y-%m-%d')
    ndia = fechas[fechas.dt.strftime('%A') == dia.lower()]
    respuesta=ndia.shape[0]
    return {'dia':dia, 'cantidad':respuesta}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score'''
    # Convertir el título a minúsculas
    titulo = titulo.lower()
    # Filtrar el DataFrame para obtener la película correspondiente al título ingresado
    pelicula = df_completo[df_completo['title'].str.lower() == titulo]
    if pelicula.empty:
        return {'mensaje': 'No se encontró ninguna filmación con ese título.'}
    anio = pelicula['release_year'].iloc[0]
    popularidad = pelicula['popularity'].iloc[0]
    return {'titulo': titulo, 'anio': anio, 'popularidad': popularidad}

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. 
    La misma variable deberá de contar con al menos 2000 valoraciones, 
    caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'''
    low = titulo.lower()
    pel = df_completo[df_completo['title'].str.lower() == low]   # Filtrar por título ingresado
    votos = pel['vote_count'].iloc[0] # Suma de los votos
    promedio =  pel['vote_average'].iloc[0] # Valor promedio de los votos
    if votos < 2000:  # Verificar si tiene menos de 2000 valoraciones
        return {'mensaje': f'La filmación "{titulo}" no cumple la condición de tener al menos 2000 valoraciones. ',"votos": votos}
    return {'titulo': titulo, 'voto_total': votos, 'voto_promedio': promedio}

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    '''Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, la cantidad de películas en las que ha participado y el promedio de retorno'''
    nombre_actor = nombre_actor.lower() # Convertir el nombre del actor a minúsculas
    peliculas_actor = df_completo[df_completo['namecast'].str.lower().str.contains(nombre_actor)] # Filtrar el DataFrame para obtener las películas en las que participa el actor
    cantidad_filmaciones = len(peliculas_actor) # Obtener la cantidad de películas y el retorno total del actor
    retorno_total = peliculas_actor['return'].sum() # Calcular el retorno promedio
    retorno_promedio = retorno_total / cantidad_filmaciones
    return {'actor': nombre_actor, 'cantidad_filmaciones': cantidad_filmaciones, 'retorno_total': retorno_total, 'retorno_promedio': retorno_promedio}


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    '''Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''
    
    nombre_director = nombre_director.lower()# Convertir el nombre del director a minúsculas
    peliculas_director = df_completo[df_completo['directors'].str.lower().str.contains(nombre_director)]# Filtrar el DataFrame para obtener las películas del director
    retorno_total_director = peliculas_director['return'].sum()# Obtener el retorno total del director
    peliculas = []  # Crear una lista para almacenar la información de cada película
    for index, pelicula in peliculas_director.iterrows():
        # Obtener los datos de cada película
        nombre_pelicula = pelicula['title']
        anio_lanzamiento = pelicula['release_year']
        retorno_pelicula = pelicula['return']
        budget_pelicula = pelicula['budget']
        revenue_pelicula = pelicula['revenue']
        
        # Agregar los datos de la película a la lista
        peliculas.append({
            'nombre': nombre_pelicula,
            'anio': anio_lanzamiento,
            'retorno_pelicula': retorno_pelicula,
            'budget_pelicula': budget_pelicula,
            'revenue_pelicula': revenue_pelicula
        })
    
    return {
        'director': nombre_director,
        'retorno_total_director': retorno_total_director,
        'peliculas': peliculas
    }


# # ML

@app.get('/recomendacion/{titulo}')
def recomendation(title: str):
    """Ingresas un nombre de pelicula y te recomienda las similares en una lista"""
    title = title.lower()
    # Buscamos la película en el df de películas
    movie_id = df_completo.loc[df_completo["title"].str.lower() == title].index
    # Si la película no existe en el DataFrame, se devuelve nulo
    if movie_id.empty:
        return {"lista recomendada": "No se encontró la película"}
    # Intentamos obtener la sinopsis de la película ingresada por el usuario
    try:
        movie_overview = df_completo.loc[movie_id, 'overview'].iloc[0]
    except KeyError:
        return {"lista recomendada": "No hay suficiente información de esta película para hacer alguna recomendación"}
    # Creamos un vectorizador TF-IDF para convertir las sinopsis en una matriz de características
    vectorizer = TfidfVectorizer()
    matriz_caracteristicas = vectorizer.fit_transform(df_completo['overview'])
    # Calculamos la similitud de coseno entre la sinopsis de la película ingresada por el usuario y todas las demás sinopsis
    cosine_sim = cosine_similarity(vectorizer.transform([movie_overview]), matriz_caracteristicas)
    # Obtenemos los 5 índices de las películas más similares excluyendo la primera que es la que ingresó el usuario
    similar_movies = cosine_sim.argsort()[0][::-1][1:6]
    recomendation_list = []
    # Obtenemos los títulos de las películas similares y los añadimos a la lista de recomendación
    for i in similar_movies:
        recomendation_list.append((df_completo["title"].iloc[i]).title())
    return {"lista recomendada": sorted(recomendation_list)}


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)