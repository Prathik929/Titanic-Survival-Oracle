# Titanic Survival Oracle

AI project: logistic regression on the Kaggle Titanic dataset + animated frontend (animated sea, live gauge, training visualisation).

## Run from VS Code
1. Open this project folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Install dependencies once: `python -m pip install -r requirements.txt`
4. Run the entire program: `python run.py`

The command trains the model from `data/train.csv`, starts the website, and opens it in your default browser at `http://127.0.0.1:8000/`. Keep the terminal open while using the website; press **Ctrl+C** to stop it.

If `python` is not recognized in the VS Code terminal, try `py -m pip install -r requirements.txt` and then `py run.py`.

## Structure
- `train.py`     Python training + export
- `run.py`       One-command training and website launcher
- `web/index.html`  animated frontend (pure HTML/CSS/JS)
- `web/model.js`    generated weights (after training)
