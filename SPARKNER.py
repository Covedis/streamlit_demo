import json
import os
import sys
import os

# import findspark
# findspark.init()
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-16.0.2"
sys.path.append("C:\\Program Files\\Java\\jdk-16.0.2")
os.environ["SPARK_HOME"]="C:\\Users\\shankar.rs\\.conda\pkgs\\pyspark-3.0.1-pyhd3eb1b0_0\\site-packages\\pyspark"
sys.path.append("C:\\Users\\shankar.rs\\.conda\pkgs\\pyspark-3.0.1-pyhd3eb1b0_0\\site-packages\\pyspark")
os.environ["HADOOP_HOME"]="C:\\Users\\shankar.rs\\OneDrive - Fractal Analytics Pvt. Ltd\\Documents\\streamlit\\winutils-master\\hadoop-3.2.1"
sys.path.append("C:\\Users\\shankar.rs\\OneDrive - Fractal Analytics Pvt. Ltd\\Documents\\streamlit\\winutils-master\\hadoop-3.2.1")
os.environ["PYSPARK_PYTHON"]="python"
# os.environ['PATH']="%PATH%;%HADOOP_HOME%/bin"
import pandas as pd
# from tqdm import tqdm
from collections import Counter
# import findspark-
import sparknlp
# findspark.init()
# findspark.find()
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.annotator.ner import ner_converter 
from sparknlp.training import CoNLL
from sparknlp.pretrained import PretrainedPipeline
import pyspark.sql.functions as F
# from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

import streamlit as st
# from sparknlp.annotator.ner.ner_converter

spark = sparknlp.start()
# spark = SparkSession.builder \
#     .appName("Spark NLP")\
#     .master("local[*]")\
#         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.7")
# spark
# # spark, sc = _initialize_spark()
st.write("[Link to Spark window](http://localhost:4040)")

st.write("## Create RDD from a Python list")


# model_save_path="streamlit\\SPARKNER_PIPELINE_13123"
model_save_path="C:\\Users\\shankar.rs\\OneDrive - Fractal Analytics Pvt. Ltd\\Documents\\streamlit\\SPARKNER_PIPELINE_13123"
sys.path.append(model_save_path)
# model_save_path="\\streamlit\\SPARKNER_PIPELINE_13123"

converter = NerConverter()\
            .setInputCols(["document","token","ner"])\
            .setOutputCol("ner_span")

cus_ner_pipeline = PipelineModel.load(model_save_path)

inp_text=st.text_input("Type or Paste the Dataset to be Extract the Entity:::!")

Input_Data = spark.createDataFrame([[inp_text]]).toDF("text")


# preds = cus_ner_pipeline.transform(Input_Data)

# prediction=converter.transform(preds)
# final_df = prediction.select(F.explode(F.arrays_zip(prediction.ner_span.result,
#                                     prediction.ner_span.metadata)).alias("entities")) \
#       .select(F.expr("entities['0']").alias("chunk"),
#               F.expr("entities['1'].entity").alias("entity"))

# output_ner = [x['chunk'] + " : " + x['entity'] for x in final_df.select(F.explode(F.arrays_zip(final_df.ner_span.result,
#                                     final_df.ner_span.metadata)).alias("entities")) \
#       .select(F.expr("entities['0']").alias("chunk"),
#               F.expr("entities['1'].entity").alias("entity")).collect()][0]


# st.write("The Extracted entities is "+ output_ner)

