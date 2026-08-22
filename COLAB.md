# Colab setup for Cayley/RoBERTa experiments

This repo includes `colab_roberta_cayley_setup.ipynb`, a bootstrap notebook intended to run on a Google Colab GPU runtime while using your project files.

## Recommended workflow

1. Put this project folder in Google Drive at:

   ```text
   MyDrive/cayley
   ```

   or push it to GitHub.

2. Open `colab_roberta_cayley_setup.ipynb` in Colab.

3. In Colab, select `Runtime > Change runtime type > GPU`.

4. Run the notebook from top to bottom.

5. Set `PROJECT_SOURCE` in the notebook:

   ```python
   PROJECT_SOURCE = "github"   # "drive", "github", or "upload"
   ```

## Private GitHub repo setup

If the GitHub repo is private, Colab needs a token to clone it.

1. Create a GitHub fine-grained personal access token with read access to the repo contents.
2. In Colab, open the key icon in the left sidebar.
3. Add a secret named:

   ```text
   GITHUB_TOKEN
   ```

4. Paste the token value and enable notebook access to the secret.

Do not paste the token directly into the notebook.

## How file access works

Colab kernels cannot directly read files from this local machine. The notebook solves that by making the project available inside the Colab VM through one of these routes:

- `drive`: mount Google Drive and use `/content/drive/MyDrive/cayley`
- `github`: clone a repository into `/content/cayley`
- `upload`: upload a zip of the project into the runtime

After setup, the notebook adds the project directory to `sys.path` and installs it in editable mode if it has a `pyproject.toml` or `setup.py`.

## Benchmark entrypoint

Once your benchmark script exists, set this in the notebook:

```python
BENCHMARK_COMMAND = "python scripts/benchmark_roberta_cayley.py"
```

Then run the final command cell.
