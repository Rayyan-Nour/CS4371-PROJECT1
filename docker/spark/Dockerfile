FROM apache/spark:3.5.3-scala2.12-java17-python3-ubuntu

USER root

# Kafka connector jars baked in, so spark-submit needs no --packages
RUN cd /opt/spark/jars && \
    for url in \
      https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.3/spark-sql-kafka-0-10_2.12-3.5.3.jar \
      https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.3/spark-token-provider-kafka-0-10_2.12-3.5.3.jar \
      https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar \
      https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar; \
    do wget -q "$url"; done

RUN pip install --no-cache-dir numpy pandas spacy kafka-python websockets requests && \
    python3 -m spacy download en_core_web_sm

ENV PYSPARK_PYTHON=python3 \
    PATH=/opt/spark/bin:$PATH \
    PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip
