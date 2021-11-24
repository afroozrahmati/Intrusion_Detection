import flwr as fl

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("172.29.80.1:8080", config={"num_rounds": 1000})