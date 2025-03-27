# Thai-Boxing-Assistant

We explored two approaches to recognize diverse strikes/kicks:

- **Dynamic Time Warping (DTW)**: Aligns motion sequences temporally using single reference samples
- **Random Forest (RF)**: Leverages feature engineering and multi-sample training for classification

Using MediaPipe, we extracted 33 body keypoints (x,y,z,visibility) to model movements through both temporal alignment and statistical learning.

## Key Findings

- **RF outperformed DTW**
- Both models faced challenges generalizing to new users due to technique variations
- DTW's "one reference sample" approach showed limited adaptability compared to RF's learned patterns
- Angle velocities and limb speeds emerged as critical features for RF

## Installation (Windows/macOS)

Python 3.12 is required due to Mediapipe

```bash
# Clone the repository
git clone https://github.com/mithuGit/Thai-Boxing-Trainer.git

# Create virtual environment
python.12 -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS:
source .venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe pandas scikit-learn fastdtw scipy numpy joblib matplotlib

#run program (please refer to the specific README for DTW or Random Forest for usage guidelines)
python3.12 .py
```

Developed by: Mithusan Naguleswaran, Nils Kovacic, Maximilian Laue, Tim Duc Minh, Ebenhaezer Aubrey Sopacua

Special Thanks to supervisor Quentin Delfosse and Vincent Scharf for their valuable insights and support in developing our ideas.
Also to the members of the Kickboxing Club at TU Darmstadt, who volunteered to be filmed for our dataset.
