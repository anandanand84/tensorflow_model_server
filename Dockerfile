FROM ubuntu
COPY tensorflow_model_server /app/
WORKDIR /app
EXPOSE 8500
ENTRYPOINT ["./tensorflow_model_server"]
