---
title: "What came first: the chicken or the egg?"
mathjax: true
bokeh: true
layout: post
excerpt_separator: <!--more-->
---

Using the datasets from <a href="https://data.buenosaires.gob.ar">Buenos Aires Datasets webpage</a> we can plot some interesting information such as population density. 


When we do that, we see that most of the population density follows the subway lines. Do the subway lines grow to reach dense areas or do densely populated areas grow around the subway lines?

<center>
{% include MapaDensidadPoblacion_div.html %}
{% include MapaDensidadPoblacion_script.html %}
</center>

<!--more-->

<h3>How can we figure out what's going on?</h3>

For example, we can look at the difference between the situation in the years $$2001$$ and $$2010$$:

<center>
{% include MapaDensidadPoblacion2001_div.html %}
{% include MapaDensidadPoblacion2001_script.html %}
</center>

<center>
{% include MapaDensidadPoblacion2010_div.html %}
{% include MapaDensidadPoblacion2010_script.html %}
</center>

From the looks of it, the areas of higher density in Buenos Aires haven't changed much in that period. And we can see how Subte A and B are moving towards big dense areas (in fact, by this date, Subte B has reached the disconnected dense cluster in Villa Urquiza). This seems to suggest that the subway is growing towards the areas of higher population density.

Why $$2001$$ and $$2010$$? The census data for those years can be found <a href="https://data.buenosaires.gob.ar/dataset/informacion-censal-por-radio">here</a>. 

The data for subway stations can be found <a href="https://data.buenosaires.gob.ar/dataset/subte-estaciones">here</a>. But a word of caution: that dataset contains the latest subway stations, so we manually removed the ones that did not exist before $$2001$$ and $$2010$$. A very detailed account of each station and the year they opened can be found in <a href="https://es.wikipedia.org/wiki/Subte_de_Buenos_Aires#Estaciones">Wikipedia</a>.

Another word of caution, if you want to replicate this: the geographic information in the $$2001$$ census data needs to be modified. As it is, the $$88$$-th row is an open polygon, which we can close "by hand". This just means that we add the first point to the end of the string, thus closing the polygon. 

If you liked the pictures and would like to see the code, you can find it on my Github page!
