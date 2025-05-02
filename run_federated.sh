#!/bin/bash

# Create logs directory if not exists
mkdir -p logs

# Start the federated server
echo "Starting Federated Server..."
python src/cli.py train-federated --model dnn --server --rounds 5 --min_clients 2 --addr [::]:8080 > logs/server.log 2>&1 &
SERVER_PID=$!

# Give the server a few seconds to start
sleep 3

# Start Client 0
echo "Starting Client 0..."
python src/cli.py train-federated --model dnn --client --client_id 0 --num_clients 2 --addr [::]:8080 --data data/client0.csv > logs/client0.log 2>&1 &
CLIENT0_PID=$!

# Start Client 1
echo "Starting Client 1..."
python src/cli.py train-federated --model dnn --client --client_id 1 --num_clients 2 --addr [::]:8080 --data data/client1.csv > logs/client1.log 2>&1 &
CLIENT1_PID=$!

# Wait for all to complete
wait $SERVER_PID
wait $CLIENT0_PID
wait $CLIENT1_PID

echo "Federated Training Completed. Check logs/server.log, client0.log, and client1.log for details."
