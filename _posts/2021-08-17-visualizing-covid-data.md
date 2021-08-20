---
title: "Visualizing COVID-19 Data"
mathjax: true
bokeh: true
layout: post
excerpt_separator: <!--more-->
---

In this post we are going to show some visualizations of COVID-19 data obtained from the <a href="https://data.buenosaires.gob.ar/dataset/casos-covid-19">Buenos Aires Datasets webpage</a>. This dataset is updated daily and contains the information of new cases and deaths in Buenos Aires, together with some extra information such as age and neighborhood.

The goal is to show how to work with this data, together with some geographical data, to obtain some nice visualizations using Bokeh.

<!--more-->

<h3> The dataset </h3>

The dataset downloaded from Buenos Aires Datasets webpage looks like this:

{% highlight python %}
import pandas as pd
import numpy as np

df = pd.read_csv("./Datasets/casos_covid19.csv")
{% endhighlight %}

|    |   numero_de_caso | fecha_apertura_snvs       | fecha_toma_muestra        | fecha_clasificacion       | provincia    | barrio           |   comuna | genero    |   edad | clasificacion   |   fecha_fallecimiento |   fallecido | fecha_alta                | tipo_contagio   |
|---:|-----------------:|:--------------------------|:--------------------------|:--------------------------|:-------------|:-----------------|---------:|:----------|-------:|:----------------|----------------------:|------------:|:--------------------------|:----------------|
|  0 |         15399546 | 24JUN2021:00:00:00.000000 | 27JUN2021:00:00:00.000000 | 27JUN2021:00:00:00.000000 | Buenos Aires | nan              |      nan | femenino  |     53 | confirmado      |                   nan |         nan | 08JUL2021:00:00:00.000000 | Comunitario     |
|  1 |         15420990 | 24JUN2021:00:00:00.000000 | 24JUN2021:00:00:00.000000 | 24JUN2021:00:00:00.000000 | CABA         | PARQUE PATRICIOS |        4 | femenino  |     61 | confirmado      |                   nan |         nan | 08JUL2021:00:00:00.000000 | Comunitario     |
|  2 |         15426848 | 24JUN2021:00:00:00.000000 | 28JUN2021:00:00:00.000000 | 28JUN2021:00:00:00.000000 | Buenos Aires | nan              |      nan | femenino  |     39 | confirmado      |                   nan |         nan | 08JUL2021:00:00:00.000000 | Comunitario     |
|  3 |         15476146 | 25JUN2021:00:00:00.000000 | 25JUN2021:00:00:00.000000 | 25JUN2021:00:00:00.000000 | Buenos Aires | nan              |      nan | masculino |     42 | confirmado      |                   nan |         nan | 08JUL2021:00:00:00.000000 | Comunitario     |
|  4 |         15494419 | 25JUN2021:00:00:00.000000 | 25JUN2021:00:00:00.000000 | 25JUN2021:00:00:00.000000 | CABA         | RECOLETA         |        2 | femenino  |     74 | confirmado      |                   nan |         nan | 08JUL2021:00:00:00.000000 | Comunitario     |

Let's say that we want to understand the number of cases and the number of deaths by date. First of all, we only keep the data of <b>confirmed</b> cases. Secondly, we need to turn the date into a readable format.

{% highlight python %}
df = df[df["clasificacion"]=="confirmado"] # Keeps only the confirmed cases
df = df.dropna(subset=["fecha_apertura_snvs"]) # Keeps only the data with date

def transform_date(date):
    decoder = {"JAN":'01',"FEB" : '02',"MAR" : '03', "APR" : '04', "MAY" : '05', "JUN" : '06', "JUL" : '07',"AUG" : '08',"SEP" : '09',"OCT" : '10',"NOV" : '11',"DEC" : '12'}
    return date[:2] + "-" + decoder[date[2:5]] + "-" + date[5:9]

df["fecha_apertura_snvs"] = df["fecha_apertura_snvs"].apply(transform_date)
df["fecha_apertura_snvs"] = pd.to_datetime(df["fecha_apertura_snvs"], format="%d-%m-%Y")
df.sort_values(by="fecha_apertura_snvs")
{% endhighlight %}

Now, we can use the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html">group by</a> statement to count the number of cases (we count on the unique case-id number).

{% highlight python %}
cases_by_date = df.groupby("fecha_apertura_snvs", as_index=False).count()
cases_by_date = cases_by_date[["fecha_apertura_snvs","numero_de_caso"]] # We keep only the dates and number of cases 
cases_by_date = cases_by_date.rename(columns={"fecha_apertura_snvs":"date", "numero_de_caso":"cases"}) # We rename those two columns to date and cases
{% endhighlight %}

To count the number of deaths, we need to filter by the column ```fallecido```, and again use the group by statement to count the number of deaths.

{% highlight python %}
deaths_by_date = df[df["fallecido"]=="si"].groupby("fecha_apertura_snvs", as_index=False).count()
deaths_by_date = deaths_by_date[["fecha_apertura_snvs","numero_de_caso"]] # We keep only the dates and number of cases 
deaths_by_date = deaths_by_date.rename(columns={"fecha_apertura_snvs":"date", "numero_de_caso":"deaths"}) # We rename those two columns to date and deaths
{% endhighlight %}

We can put this data together by <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html?highlight=merge#pandas.merge">merging</a> the two tables

{% highlight python %}
data_by_date = pd.merge(cases_by_date,deaths_by_date,on="date")
{% endhighlight %}

Finally, we can add a <i>7-day moving average</i>. This will help us better understand the tendency by smoothing out the daily variation we'll see in the plot.

{% highlight python %}
data_by_date["casesMA"] = data_by_date["cases"].rolling(window=7).mean()
data_by_date["deathsMA"] = data_by_date["deaths"].rolling(window=7).mean()
{% endhighlight %}

We have all that we need to create our first plot! This is obtained by using <a href="https://bokeh.org/">BOKEH visualization library</a> for Python.

We can plot the number of cases by date

<center>
{% include covid_cases_by_date_div.html %}
{% include covid_cases_by_date_script.html %}
</center>

and the number of deaths by date

<center>
{% include covid_deaths_by_date_div.html %}
{% include covid_deaths_by_date_script.html %}
</center>

Clicking on the legend allows you to hide the plot.

<h3> Geographic datasets </h3>

Just like we use Pandas to analyze datasets, we can use <a href="https://geopandas.org/">GeoPandas</a> to analyze geographic datasets. Moreover, we can find a map of Buenos Aires together with its division into <a href="https://data.buenosaires.gob.ar/dataset/barrios">neighborhoods</a> and <a href="https://data.buenosaires.gob.ar/dataset/informacion-censal-por-radio">census data by area</a> in the Buenos Aires Data webpage.

How do these geographic datasets look like? Essentially like a usual Pandas dataset, with an extra ```geometry``` column, which will be used to draw the shape of our data.

{% highlight python %}
import geopandas as gpd

neighborhoods = gpd.read_file("./Datasets/barrios.geojson")
census = gpd.read_file("./Datasets/caba_radios_censales.geojson")
{% endhighlight %}

|    | barrio           |   comuna |   perimetro |        area | geometry                                                                                                                                                                                                                                                          |
|---:|:-----------------|---------:|------------:|------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | CHACARITA        |       15 |     7724.85 | 3.11571e+06 | POLYGON ((-58.4528200492791 -34.5959886570639, ... ))                                                                                                                                                                                           |
|  1 | PATERNAL         |       15 |     7087.51 | 2.22983e+06 | POLYGON ((-58.4655768128541 -34.5965577078058, ... ))                                                                                                                                                                                                                                                                |
|  2 | VILLA CRESPO     |       15 |     8131.86 | 3.61598e+06 | POLYGON ((-58.4237529813037 -34.5978273383243, ... ))                                                                                                                                                                                                                               |
|  3 | VILLA DEL PARQUE |       11 |     7705.39 | 3.3996e+06  | POLYGON ((-58.4946097568899 -34.6148652395239, ... )) |
|  4 | ALMAGRO          |        5 |     8537.9  | 4.05075e+06 | POLYGON ((-58.4128700313089 -34.6141162515854, ... ))                                                                                                                                                                                                                                                                                                  |

As you can see, the column ```geometry``` contains the necessary latitude-longitude coordinates to plot the points corresponding to the shape of each neighborhood.

Now we will aggregate our original table by neighborhood and merge it together with this table, so that we keep the number of cases by neighborhood and the shape of each neighborhood in the same table. Also, we will be able to find out the population of each neighborhood from the census data.

{% highlight python %}

population_by_neighborhood = census[["BARRIO","POBLACION"]].groupby("BARRIO").sum()
neighborhoods = neighborhoods.join(on="barrio", other=population_by_neighborhood) # Aggregate data of population by neighborhood

neighborhoods = neighborhoods.join(on="barrio",other=df.dropna(subset=["barrio"]).groupby("barrio").count()["numero_de_caso"]) # Aggregate data of cases by neighborhood
neighborhoods = neighborhoods.join(on="barrio",other=df.dropna(subset=["barrio","fallecido"]).groupby("barrio").count()["fallecido"]) # Aggregate data of deaths by neighborhood

neighborhoods["case_density"] = neighborhoods["numero_de_caso"]/neighborhoods["POBLACION"]
neighborhoods["death_density"] = neighborhoods["fallecido"]/neighborhoods["POBLACION"]

neighborhoods = neighborhoods.rename(columns={"barrio":"neighborhood", "numero_de_caso":"cases", "POBLACION":"population", "fallecido":"deaths"}) # Rename the columns for clarity
{% endhighlight %}

This is sufficient information to display the information on the map.

The following two maps show the number of covid cases per 1000 people in each neighborhood, and of deaths per 1000 people in each neighborhood.

<center>
{% include covid_cases_by_neighborhood_div.html %}
{% include covid_cases_by_neighborhood_script.html %}
</center>


<center>
{% include covid_deaths_by_neighborhood_div.html %}
{% include covid_deaths_by_neighborhood_script.html %}
</center>

Finally, we can also plot the number of new cases and deaths in each neighborhood by month, and see how it changes with time. For this, we will need to play with our data a bit.

We can keep track of the month and year, and aggregate the number of cases corresponding to each neighborhood during each period using the <a href="https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html">pivot table function</a>.

{% highlight python %}
df["month-year"] = df["fecha_apertura_snvs"].apply(lambda x: x.month_name()[:3]+str(x.year)) # Keeps track of month and year
df["cases"] = df["numero_de_caso"].apply(lambda x:1) # Artificial column: made to aggregate with pivot_table
monthly_data = df.pivot_table(index="barrio", columns ="month-year", values = "cases", aggfunc=np.sum)
monthly_data = monthly_data.reset_index()
{% endhighlight %}

We obtain in this way a table like this

| barrio            |   Apr2020 |   Apr2021 |   Aug2020 |   Aug2021 |   Dec2020 |   Feb2021 |   Jan2021 |   Jul2020 |   Jul2021 |   Jun2020 |   Jun2021 |   Mar2020 |   Mar2021 |   May2020 |   May2021 |   Nov2020 |   Oct2020 |   Sep2020 |
|:------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| AGRONOMIA         |         4 |       285 |       100 |        55 |        57 |        91 |       156 |        70 |        95 |        28 |       167 |         3 |       114 |         5 |       252 |        38 |        72 |       115 |
| ALMAGRO           |        45 |      3373 |      1607 |       306 |       727 |      1009 |      1578 |      1391 |      1142 |       716 |      1774 |        17 |      1364 |       179 |      3261 |       548 |       987 |      1456 |
| BALVANERA         |        54 |      3955 |      2297 |       319 |       757 |       938 |      1590 |      2271 |      1156 |      1326 |      2090 |        42 |      1549 |       328 |      3954 |       575 |      1061 |      1784 |
| BARRACAS          |        23 |      2674 |      1617 |       148 |       491 |       479 |       824 |      1825 |       653 |      1883 |      1380 |         3 |       858 |       470 |      2605 |       275 |       504 |       992 |
| BELGRANO          |        60 |      3008 |      1065 |       418 |       875 |      1011 |      2089 |       852 |      1057 |       274 |      1568 |        30 |      1463 |       110 |      2582 |       477 |       848 |      1127 |


where the column for each period has the number of cases in the corresponding neighborhood.

We can now merge this table with the ```neighborhoods``` one, so that we have the number of cases by month for each neighborhood, together with the geographical data.

{% highlight python %}
cases_by_month = neighborhoods.merge(monthly_data)
{% endhighlight %}

This is all we need to do our final plot.

<center>
{% include covid_monthly_cases_div.html %}
{% include covid_monthly_cases_script.html %}
</center>

There're still a lot of things we can do with these datasets and these tools: we can visualize the histogram of age for COVID-19 cases or deaths in each neighborhood, we can look at the correlation between cases or deaths and population density (since we know the area of each neighborhood is in the dataset!), just to name a few.

If you liked the plots and would like to see the code, you can find it on my Github page!
