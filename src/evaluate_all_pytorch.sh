echo "Starting evaluation ..."

# bash evaluate_pytorch.sh 60 3 0.00625 0.96 1040 && \
# bash evaluate_pytorch.sh 60 5 0.00625 0.96 1040 && \
# bash evaluate_pytorch.sh 62 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 62 5 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 63 3 0.003125 0.96 1040 && \
# bash evaluate_pytorch.sh 63 5 0.003125 0.96 1040 && \
# bash evaluate_pytorch.sh 64 3 0.00625 0.96 1040 && \
# bash evaluate_pytorch.sh 64 5 0.00625 0.96 1040 && \
# bash evaluate_pytorch.sh 65 7 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 66 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 66 5 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 67 3 0.0015625 0.96 1040 && \
# bash evaluate_pytorch.sh 67 7 0.0015625 0.96 1040 && \
# bash evaluate_pytorch.sh 68 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 68 7 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 69 3 0.0015625 0.96 1040 && \
# bash evaluate_pytorch.sh 69 7 0.0015625 0.96 1040 && \
# bash evaluate_pytorch.sh 70 3 0.05 0.96 1040 && \
# bash evaluate_pytorch.sh 70 9 0.05 0.96 1040 && \
# bash evaluate_pytorch.sh 61 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 61 5 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 71 3 0.05 0.96 1040 && \
# bash evaluate_pytorch.sh 71 9 0.05 0.96 1040

# bash evaluate_pytorch.sh 72 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 72 9 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 73 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 73 5 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 74 5 0.05 0.96 1040 && \
# bash evaluate_pytorch.sh 75 3 0.05 0.96 1040 && \
# bash evaluate_pytorch.sh 75 5 0.05 0.96 1040 && \
# bash evaluate_pytorch.sh 76 3 0.025 0.96 1040 && \
# bash evaluate_pytorch.sh 76 5 0.025 0.96 1040

# bash evaluate_pytorch.sh 78 5 0.0001 0.99 1040 && \
# bash evaluate_pytorch.sh 80 5 0.001 0.5 1040 && \
# bash evaluate_pytorch.sh 77 5 0.01 0.95 1040 && \
# bash evaluate_pytorch.sh 79 5 0.025 0.96 1040

# bash evaluate_pytorch.sh 81 9 0.05 0.96 1040

# bash evaluate_pytorch.sh 500 3 0.0125 0.96 200 && \
# bash evaluate_pytorch.sh 500 3 0.00625 0.96 200 && \
# bash evaluate_pytorch.sh 500 3 0.003125 0.96 200 && \
# bash evaluate_pytorch.sh 501 3 0.025 0.96 200 && \
# bash evaluate_pytorch.sh 502 3 0.01 0.96 200 && \
# bash evaluate_pytorch.sh 504 3 0.00625 0.96 200 && \
# bash evaluate_pytorch.sh 504 3 0.003125 0.96 200 && \
# bash evaluate_pytorch.sh 504 3 0.0015625 0.96 200 && \
# bash evaluate_pytorch.sh 505 3 0.0125 0.96 200 && \
# bash evaluate_pytorch.sh 505 3 0.00625 0.96 200 && \
# bash evaluate_pytorch.sh 505 3 0.003125 0.96 200 && \
# bash evaluate_pytorch.sh 503 3 0.025 0.96 200 && \
# bash evaluate_pytorch.sh 503 3 0.0125 0.96 200 && \
# bash evaluate_pytorch.sh 503 3 0.00625 0.96 200 && \
# bash evaluate_pytorch.sh 506 3 0.05 0.96 200 && \
# bash evaluate_pytorch.sh 506 3 0.025 0.96 200 && \
# bash evaluate_pytorch.sh 506 3 0.0125 0.96 200


bash evaluate_pytorch.sh 82 2 0.003125 0.96 1040 && \
bash evaluate_pytorch.sh 86 2 0.0015625 0.96 1040 && \
bash evaluate_pytorch.sh 87 2 0.003125 0.96 1040 && \
bash evaluate_pytorch.sh 85 2 0.0125 0.96 1040 && \
bash evaluate_pytorch.sh 88 2 0.0125 0.96 1040 && \
bash evaluate_pytorch.sh 82 4 0.003125 0.96 1040 && \
bash evaluate_pytorch.sh 83 2 0.01 0.96 1040


# for PROJECT_INDEX in 1
# do
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.1 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.05 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.025 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.0125 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.00625 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.003125 0.96 200
    # # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.0015625 0.96 200
# done