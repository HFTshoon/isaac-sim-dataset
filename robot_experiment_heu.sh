
# scene01 0 1 2 3 4
# scene02 0 1 5 7 8

# heuristic
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 000 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 001 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 002 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 003 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 005 --headless

./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 000 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 001 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 005 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 007 --headless
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 008 --headless


./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 000 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 001 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 002 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 003 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 01 --scene-num 005 --headless --jitter

./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 000 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 001 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 005 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 007 --headless --jitter
./python.sh corl2025/scenes/experiment_heu.py --scene 02 --scene-num 008 --headless --jitter

