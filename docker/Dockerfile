FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b11347
# Python 3 is already provided by this pinned upstream image.
WORKDIR /app
COPY adapter.py /opt/sagemaker/adapter.py
COPY entrypoint.sh /opt/sagemaker/entrypoint.sh
RUN chmod 755 /opt/sagemaker/entrypoint.sh && python3 -m py_compile /opt/sagemaker/adapter.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["/opt/sagemaker/entrypoint.sh"]
CMD ["serve"]
