# Streamlit Demo

Run the full pipeline first so local fitted artifacts exist:

```powershell
python -m src.run_pipeline
```

Then launch the app:

```powershell
streamlit run app/scorecard_app.py
```

The app lets a reviewer enter borrower characteristics and see:

- predicted probability of default
- credit score
- simple risk band
- top model reason codes
- scorecard points table

The app uses ignored local model artifacts from `models/`, so those artifacts
are regenerated locally rather than committed to GitHub.

## Standalone HTML App

The project also generates a browser-only version at:

```text
app/scorecard_app.html
```

Open that file directly in a browser after running the pipeline. It embeds the
current scorecard points and model coefficients, so regenerate it with:

```powershell
python -m src.export_static_app
```
