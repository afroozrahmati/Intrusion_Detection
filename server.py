import flwr as fl

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("192.168.1.237:8080", config={"num_rounds": 1000})